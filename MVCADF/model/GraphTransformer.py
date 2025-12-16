import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    BCEWithLogitsLoss,
    Linear,
    Sequential,
    ReLU,
)
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GCN2Conv,
    GATConv,
    ECConv,
    global_mean_pool,
    GINConv,
)
from torch_geometric.nn.conv import MessagePassing
from collections import OrderedDict
from torch_geometric.utils import dropout_adj

###############################################################################
# 1. GraphTransformer相关类
###############################################################################
class SpatialEncoding(nn.Module):
    def __init__(self, dim_model):
        super(SpatialEncoding, self).__init__()
        self.dim = dim_model
        self.fnn = Sequential(
            Linear(1, dim_model),
            ReLU(),
            Linear(dim_model, 1),
            ReLU()
        )

    def reset_parameters(self):
        self.fnn[0].reset_parameters()
        self.fnn[2].reset_parameters()

    def forward(self, lap):
        lap_ = torch.unsqueeze(lap, dim=-1)
        out = self.fnn(lap_)
        return out

class MultiheadAttention(MessagePassing):
    def __init__(self, dim_model, num_heads, rel_encoder, spatial_encoder, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.d_model = dim_model
        self.num_heads = num_heads

        self.rel_embedding = rel_encoder
        self.rel_encoding = Sequential(
            Linear(dim_model, 1),
            ReLU()
        )
        self.spatial_encoding = spatial_encoder
        assert dim_model % num_heads == 0
        self.depth = self.d_model // num_heads

        self.wq = Linear(dim_model, dim_model)
        self.wk = Linear(dim_model, dim_model)
        self.wv = Linear(dim_model, dim_model)
        self.dense = Linear(dim_model, dim_model)

    def reset_parameters(self):
        self.rel_embedding.reset_parameters()
        self.rel_encoding[0].reset_parameters()
        self.spatial_encoding.reset_parameters()
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.dense.reset_parameters()

    def denominator(self, qs, ks):
        all_ones = torch.ones([ks.shape[0]], device=qs.device)
        ks_sum = torch.einsum("nhm,n->hm", ks, all_ones)
        return torch.einsum("nhm,hm->nh", qs, ks_sum)

    def forward(self, x, sp_edge_index, sp_value, edge_rel):
        rel_embedding = self.rel_embedding(edge_rel)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x).view(x.shape[0], self.num_heads, self.depth)

        row, col = sp_edge_index
        query_end = q[col] + rel_embedding
        key_start = k[row] + rel_embedding

        query_end = query_end.view(sp_edge_index.shape[1], self.num_heads, self.depth)
        key_start = key_start.view(sp_edge_index.shape[1], self.num_heads, self.depth)

        edge_attn_num = torch.einsum("ehd,ehd->eh", query_end, key_start)
        data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(edge_attn_num.shape[-1], dtype=torch.float32)))
        edge_attn_num *= data_normalizer
        edge_attn_bias = self.spatial_encoding(sp_value)
        edge_attn_num += edge_attn_bias.squeeze(-1).unsqueeze(-1)

        attn_normalizer = self.denominator(
            q.view(x.shape[0], self.num_heads, self.depth),
            k.view(x.shape[0], self.num_heads, self.depth)
        )
        edge_attn_dem = attn_normalizer[col]
        attention_weight = edge_attn_num / edge_attn_dem

        outputs = []
        for i in range(self.num_heads):
            output_per_head = self.propagate(
                edge_index=sp_edge_index,
                x=v[:, i, :],
                edge_weight=attention_weight[:, i],
                size=None
            )
            outputs.append(output_per_head)
        out = torch.cat(outputs, dim=-1)
        return self.dense(out), attention_weight

    def message(self, x_j, edge_weight):
        return edge_weight.unsqueeze(-1)*x_j

class GraphTransformerEncode(nn.Module):
    def __init__(self, num_heads, in_dim, dim_forward, rel_encoder, spatial_encoder, dropout):
        super(GraphTransformerEncode, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.dim_forward = dim_forward

        self.ffn = Sequential(
            Linear(in_dim, dim_forward),
            ReLU(),
            Linear(dim_forward, in_dim)
        )
        self.multiHeadAttention = MultiheadAttention(
            dim_model=in_dim,
            num_heads=num_heads,
            rel_encoder=rel_encoder,
            spatial_encoder=spatial_encoder
        )
        self.layernorm1 = nn.LayerNorm(in_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(in_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def reset_parameters(self):
        self.ffn[0].reset_parameters()
        self.ffn[2].reset_parameters()
        self.multiHeadAttention.reset_parameters()
        self.layernorm1.reset_parameters()
        self.layernorm2.reset_parameters()

    def forward(self, feature, sp_edge_index, sp_value, edge_rel):
        x_norm = self.layernorm1(feature)
        attn_output, attn_weight = self.multiHeadAttention(x_norm, sp_edge_index, sp_value, edge_rel)
        attn_output = self.dropout1(attn_output)
        out1 = attn_output + feature

        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output)
        out2 = out1 + ffn_output
        return out2, attn_weight

class GraphTransformer(nn.Module):
    def __init__(
        self,
        layer_num=3,
        embedding_dim=64,
        num_heads=4,
        num_rel=10,
        dropout=0.2,
        gtype='graph'
    ):
        super(GraphTransformer, self).__init__()
        self.gtype = gtype
        self.rel_encoder = nn.Embedding(num_rel, embedding_dim)
        self.spatial_encoder = SpatialEncoding(embedding_dim)
        self.encoder = nn.ModuleList()
        for i in range(layer_num):
            self.encoder.append(
                GraphTransformerEncode(
                    num_heads=num_heads,
                    in_dim=embedding_dim,
                    dim_forward=embedding_dim*2,
                    rel_encoder=self.rel_encoder,
                    spatial_encoder=self.spatial_encoder,
                    dropout=dropout
                )
            )

    def reset_parameters(self):
        for e in self.encoder:
            e.reset_parameters()

    def forward(self, feature, data):
        x = feature
        attn_layer=[]
        for graphEncoder in self.encoder:
            if not hasattr(data, 'sp_edge_index') or data.sp_edge_index is None or data.sp_edge_index.numel() == 0:
                attn = torch.zeros(x.size(0), self.encoder[0].num_heads, device=x.device)
            else:
                x, attn = graphEncoder(x, data.sp_edge_index, data.sp_value, data.sp_edge_rel)
            attn_layer.append(attn)

        if self.gtype=='graph':
            from torch_geometric.nn import global_mean_pool
            representation = global_mean_pool(x, batch=data.batch)
            sub_representation = []
            if hasattr(data, "batch") and data.batch is not None:
                num_graphs = int(data.batch.max().item()) + 1
                for idx in range(num_graphs):
                    mask = (data.batch == idx).nonzero().flatten()
                    sub_embedding = x[mask]
                    sub_representation.append(sub_embedding)
            else:
                sub_representation.append(x)
        else:
            representation = x[data.id.nonzero().flatten()] if hasattr(data, 'id') else x
            sub_representation = []
            if data is None:
                sub_representation.append(x)
            elif hasattr(data, "batch") and data.batch is not None:
                num_graphs = int(data.batch.max().item()) + 1
                for idx in range(num_graphs):
                    mask = (data.batch == idx).nonzero().flatten()
                    sub_embedding = x[mask]
                    sub_representation.append(sub_embedding)
            else:
                sub_representation.append(x)

        return representation, sub_representation, attn_layer

###############################################################################
# 2. 跨视图对比损失函数
###############################################################################
def cross_view_contrastive_loss(view1, view2, temperature=0.5):
    view1_norm = F.normalize(view1, p=2, dim=1)
    view2_norm = F.normalize(view2, p=2, dim=1)
    sim_matrix = torch.matmul(view1_norm, view2_norm.t()) / temperature
    batch_size = sim_matrix.size(0)
    labels = torch.arange(batch_size).to(view1.device)
    loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)
    loss = loss / 2.0
    return loss

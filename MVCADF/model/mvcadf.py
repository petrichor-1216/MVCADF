# ====================== tiger.py (five-view aligned) ======================
# -*- coding: utf-8 -*-
import os, math, torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear, ReLU

from model.GraphTransformer import GraphTransformer    # 依赖 GraphTransformer.py

# -------------------------------------------------------------------------
def init_params(module, layers=2):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(0.0, 0.02 / math.sqrt(layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(0.0, 0.02)

def pair_alignment_loss(z1, z2, tau=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = torch.matmul(z1, z2.T) / tau
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(sim, labels)

def multiview_center_loss(view_list, tau=0.1):   # 五通道中心对齐
    """
    view_list: List[Tensor] – 每个 tensor 形状一致 (B, d)
    对每个视图与中心向量做对齐损失，返回标量
    """
    if len(view_list) == 0:
        return 0.0
    center = torch.stack(view_list, dim=0).mean(dim=0)  # (B, d)
    loss = 0.0
    for v in view_list:
        loss += pair_alignment_loss(v, center, tau)
    return loss
# -------------------------------------------------------------------------

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1, self.conv2 = GCNConv(in_dim, hidden_dim), GCNConv(hidden_dim, hidden_dim)
    def reset_parameters(self):
        self.conv1.reset_parameters(); self.conv2.reset_parameters()
    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index)); x = self.conv2(x, edge_index)
        return x.mean(dim=0, keepdim=True) if batch is None else global_mean_pool(x, batch)

class MotifFeatures(nn.Module):
    def __init__(self, d_model): super().__init__(); self.enc = nn.Linear(2, d_model)
    def reset_parameters(self): self.enc.reset_parameters()
    def forward(self, data):
        if data is None or not hasattr(data,'x'):
            return torch.zeros((1, self.enc.out_features), device=self.enc.weight.device)
        return self.enc(data.x)

class NodeFeatures(nn.Module):
    def __init__(self, degree, feat_dim, d_model, layer=2, gtype='graph'):
        super().__init__()
        self.node_enc = Linear(feat_dim, d_model) if gtype=='graph' else nn.Embedding(feat_dim, d_model)
        self.deg_enc  = nn.Embedding(degree, d_model, padding_idx=0)
        self.apply(lambda m: init_params(m, layers=layer))
    def reset_parameters(self): self.node_enc.reset_parameters(); self.deg_enc.reset_parameters()
    def forward(self, data):
        row,col = data.edge_index
        deg = degree(col, data.x.size(0), dtype=data.x.dtype)
        return self.node_enc(data.x) + self.deg_enc(deg.long())

class Discriminator(nn.Module):
    def __init__(self, d): super().__init__(); self.f_k = nn.Bilinear(d, d, 1); self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.f_k.weight); self.f_k.bias.data.zero_()
    def forward(self,c,h_pl,h_mi):
        return torch.cat((self.f_k(h_pl,c), self.f_k(h_mi,c)), dim=0)

class DynamicFuser(nn.Module):
    def __init__(self, d_model, T=1):
        super().__init__(); self.energy = nn.Linear(d_model,1); self.T=T
    def forward(self, feats):                         # feats list[B,d]
        e = torch.cat([self.energy(z) for z in feats], dim=1)  # (B,n)
        w = F.softmax(-e/self.T, dim=1)
        fused = torch.zeros_like(feats[0])
        for i,z in enumerate(feats): fused += w[:,i:i+1]*z
        return fused, w

class FPEncoder(nn.Module):
    def __init__(self, fp_len, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(Linear(fp_len,d_model), ReLU(),
                                  nn.Dropout(dropout),
                                  Linear(d_model,d_model))
    def forward(self, fp): return self.proj(fp.float())

# -------------------------------------------------------------------------
class MVCADF(nn.Module):
    def __init__(self, max_layer=3, num_features_drug=78, num_nodes=200,
                 num_relations_mol=10, num_relations_graph=10,
                 output_dim=64, max_degree_graph=100, max_degree_node=100,
                 sub_coeff=0.2, mi_coeff=0.5, dropout=0.2, device='cuda',
                 element_feature_dim=20, fp_len=2048):
        super().__init__()
        print("MVCADF loaded  — DynamicFusion + 1-D FP + Five-view alignment")
        self.device, self.output_dim, self.mol_coeff, self.mi_coeff = device, output_dim, sub_coeff, mi_coeff
        self.fp_len = fp_len

        # --- 节点级编码器 ---
        self.mol_atom_feature  = NodeFeatures(max_degree_graph, num_features_drug, output_dim, gtype='graph')
        self.drug_node_feature = NodeFeatures(max_degree_node,  num_nodes,         output_dim, gtype='node')
        self.motif_feature     = MotifFeatures(output_dim)
        self.fp_encoder        = FPEncoder(fp_len, output_dim, dropout)
        self.element_encoder   = GCNEncoder(element_feature_dim, output_dim)

        # --- 表示学习 ---
        self.mol_representation  = GraphTransformer(max_layer, output_dim, 4, num_relations_mol,   dropout, 'graph')
        self.node_representation = GraphTransformer(max_layer, output_dim, 4, num_relations_graph, dropout, 'node')
        self.motif_representation= GraphTransformer(2,         output_dim, 2, 1,                   dropout, 'graph')

        # --- 动态融合 ---
        self.fuser = DynamicFuser(output_dim)

        # --- 投影与分类 ---
        self.drug_proj  = nn.Sequential(Linear(output_dim,256), ReLU(), nn.Dropout(dropout),
                                        Linear(256,output_dim))
        self.classifier = nn.Sequential(Linear(output_dim*3,256), ReLU(), nn.Dropout(dropout),
                                        Linear(256,512), ReLU(), nn.Dropout(dropout),
                                        Linear(512,2))

        # --- MI 对比 ---
        self.disc = Discriminator(output_dim)
        self.b_xent = nn.BCEWithLogitsLoss()

    # ---------------------------------------------------------------------
    def reset_parameters(self):
        self.mol_atom_feature.reset_parameters()
        self.drug_node_feature.reset_parameters()
        self.motif_feature.reset_parameters()
        self.element_encoder.reset_parameters()
        self.fp_encoder.proj[0].reset_parameters(); self.fp_encoder.proj[3].reset_parameters()
        self.mol_representation.reset_parameters(); self.node_representation.reset_parameters()
        self.motif_representation.reset_parameters()
        self.fuser.energy.reset_parameters()
        for m in self.drug_proj:
            if hasattr(m,'reset_parameters'): m.reset_parameters()
        for m in self.classifier:
            if hasattr(m,'reset_parameters'): m.reset_parameters()
        self.disc.reset_parameters()
    # ---------------------------------------------------------------------

    # --- MI 辅助 ---
    def MI(self, c, sub_list):
        idx = torch.arange(c.size(0)-1,-1,-1,device=c.device)
        shuffle = c[idx]; c0,c1,subs=[],[],[]
        for a,b,s in zip(c,shuffle,sub_list):
            c0.append(a.expand_as(s)); c1.append(b.expand_as(s)); subs.append(s)
        return self.disc(torch.cat(subs), torch.cat(c0), torch.cat(c1))
    def loss_MI(self, logits):
        half = logits.size(0)//2
        lbl = torch.cat([torch.ones(half), torch.zeros(half)]).to(self.device)
        return self.b_xent(logits.view(1,-1), lbl.view(1,-1))

    # ---------------------------------------------------------------------
    def forward(self, dm1, dg1, dm2, dg2, mt1=None, mt2=None, el1=None, el2=None):

        # ---- 初始特征 ----
        mol1_feat = self.mol_atom_feature(dm1); mol2_feat = self.mol_atom_feature(dm2)
        node1_feat= self.drug_node_feature(dg1); node2_feat= self.drug_node_feature(dg2)
        motif1_feat= self.motif_feature(mt1);   motif2_feat= self.motif_feature(mt2)

        fp1_emb = self.fp_encoder(dm1.fp.view(-1, self.fp_len))
        fp2_emb = self.fp_encoder(dm2.fp.view(-1, self.fp_len))

        # ---- 图级表示 ----
        mol1_g,mol1_sub,_ = self.mol_representation (mol1_feat, dm1)
        mol2_g,mol2_sub,_ = self.mol_representation (mol2_feat, dm2)
        node1_g,node1_sb,_= self.node_representation(node1_feat, dg1)
        node2_g,node2_sb,_= self.node_representation(node2_feat, dg2)
        motif1_g,_,_ = self.motif_representation(motif1_feat, mt1) if mt1 else (torch.zeros_like(mol1_g),None,None)
        motif2_g,_,_ = self.motif_representation(motif2_feat, mt2) if mt2 else (torch.zeros_like(mol2_g),None,None)
        elem1_g = self.element_encoder(el1.x, el1.edge_index, getattr(el1,'batch',None)) if el1 and hasattr(el1,'x') \
                  else torch.zeros_like(mol1_g)
        elem2_g = self.element_encoder(el2.x, el2.edge_index, getattr(el2,'batch',None)) if el2 and hasattr(el2,'x') \
                  else torch.zeros_like(mol2_g)

        # ---- 动态融合 ----
        d1_fused,_ = self.fuser([node1_g, mol1_g, motif1_g, elem1_g, fp1_emb])
        d2_fused,_ = self.fuser([node2_g, mol2_g, motif2_g, elem2_g, fp2_emb])

        # ---- Drug 表征 & 分类 ----
        drug1_emb = self.drug_proj(d1_fused); drug2_emb = self.drug_proj(d2_fused)
        interaction = drug1_emb * drug2_emb
        score = self.classifier(torch.cat([drug1_emb, drug2_emb, interaction], dim=-1))

        # ---- Loss 计算 ----
        loss_label = F.nll_loss(F.log_softmax(score,dim=-1), dm1.y.view(-1))
        loss_m = self.loss_MI(self.MI(drug1_emb,mol1_sub)) + self.loss_MI(self.MI(drug2_emb,mol2_sub))
        loss_n = self.loss_MI(self.MI(drug1_emb,node1_sb)) + self.loss_MI(self.MI(drug2_emb,node2_sb))

        # ★ 五通道中心对齐损失
        views1 = [mol1_g, node1_g, motif1_g, elem1_g, fp1_emb]
        views2 = [mol2_g, node2_g, motif2_g, elem2_g, fp2_emb]
        loss_align = multiview_center_loss(views1) + multiview_center_loss(views2)

        loss = loss_label + self.mol_coeff*loss_m + self.mi_coeff*loss_n + 0.1*loss_align

        prob = torch.exp(F.log_softmax(score, dim=-1))[:,1]
        return prob, loss

    # ---------------------------------------------------------------------
    def save(self, path):
        torch.save(self.state_dict(), os.path.join(path, self.__class__.__name__ + '.pt'))

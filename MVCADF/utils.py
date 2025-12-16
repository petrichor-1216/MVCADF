import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Batch, Data


###############################################################################
# DTADataset
###############################################################################
class DTADataset(InMemoryDataset):
    def __init__(self, x=None, y=None,
                 sub_graph=None, smile_graph=None,
                 motif_graph=None, element_graph=None,
                 element_feature_dim=64,
                 fp_len=2048
                 ):
        super(DTADataset, self).__init__()
        self.labels = y
        self.drug_ID = x
        self.sub_graph = sub_graph
        self.smile_graph = smile_graph
        self.motif_graph = motif_graph
        self.element_graph = element_graph
        self.element_feature_dim = element_feature_dim
        self.fp_len = fp_len

        # ------------------------------------------------------------------

    def read_drug_info(self, drug_id, labels):

        (c_size, features, edge_index, rel_index,
         sp_edge_index, s_value, s_rel,
         deg, fp_vec) = self.smile_graph[str(drug_id)]

        (subset, subg_edge_idx, subg_rel, mapping_id,
         s_edge_idx2, s_value2, s_rel2,
         deg2) = self.sub_graph[str(drug_id)]

        # ---------------------- MG 图 ----------------------
        data_mol = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t(),
            y=torch.tensor([labels], dtype=torch.long),
            rel_index=torch.tensor(rel_index, dtype=torch.long),
            sp_edge_index=torch.tensor(sp_edge_index, dtype=torch.long).t(),
            sp_value=torch.tensor(s_value, dtype=torch.float),
            sp_edge_rel=torch.tensor(s_rel, dtype=torch.long),
            fp=torch.tensor(fp_vec, dtype=torch.float)
        )
        data_mol.__setitem__('c_size', torch.tensor([c_size], dtype=torch.long))

        # ---------------------- BKG 子图 ----------------------
        data_graph = Data(
            x=torch.tensor(subset, dtype=torch.long),
            edge_index=torch.tensor(subg_edge_idx, dtype=torch.long).t(),
            y=torch.tensor([labels], dtype=torch.long),
            id=torch.tensor(mapping_id, dtype=torch.bool),
            rel_index=torch.tensor(subg_rel, dtype=torch.long),
            sp_edge_index=torch.tensor(s_edge_idx2, dtype=torch.long).t(),
            sp_value=torch.tensor(s_value2, dtype=torch.float),
            sp_edge_rel=torch.tensor(s_rel2, dtype=torch.long)
        )

        return data_mol, data_graph

    # ------------------------------------------------------------------
    def read_motif_info(self, drug_id, labels):
        if (self.motif_graph is None) or (str(drug_id) not in self.motif_graph):
            return None
        data = self.motif_graph[str(drug_id)]
        data.y = torch.tensor([labels], dtype=torch.long)

        if not hasattr(data, 'sp_edge_index'):
            data.sp_edge_index = data.edge_index
            data.sp_value = torch.ones(data.edge_index.size(1), dtype=torch.float)
            data.sp_edge_rel = torch.zeros(data.edge_index.size(1), dtype=torch.long)
        return data

    # ------------------------------------------------------------------
    def read_element_info(self, drug_id, label):
        if (self.element_graph is None) or (str(drug_id) not in self.element_graph):
            return Data(
                x=torch.zeros((1, self.element_feature_dim), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0,), dtype=torch.long),
                id=torch.ones(1, dtype=torch.bool),
                y=torch.tensor([label], dtype=torch.long)
            )
        data = self.element_graph[str(drug_id)]
        data.y = torch.tensor([label], dtype=torch.long)
        return data

    # ------------------- Dataset 接口 -------------------
    def __len__(self):
        return len(self.drug_ID)

    def __getitem__(self, idx):
        drug1_id = self.drug_ID[idx, 0]
        drug2_id = self.drug_ID[idx, 1]
        label = int(self.labels[idx])

        drug1_mol, drug1_subgraph = self.read_drug_info(drug1_id, label)
        drug2_mol, drug2_subgraph = self.read_drug_info(drug2_id, label)
        drug1_motif = self.read_motif_info(drug1_id, label)
        drug2_motif = self.read_motif_info(drug2_id, label)
        drug1_element = self.read_element_info(drug1_id, label)
        drug2_element = self.read_element_info(drug2_id, label)

        return (drug1_mol, drug1_subgraph,
                drug2_mol, drug2_subgraph,
                drug1_motif, drug2_motif,
                drug1_element, drug2_element)


def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def collate(data_list):
    d1_mol_list, d1_sub_list = [], []
    d2_mol_list, d2_sub_list = [], []
    d1_motif_list, d2_motif_list = [], []
    d1_element_list, d2_element_list = [], []

    for item in data_list:
        d1_mol_list.append(item[0])
        d1_sub_list.append(item[1])
        d2_mol_list.append(item[2])
        d2_sub_list.append(item[3])
        d1_motif_list.append(item[4])
        d2_motif_list.append(item[5])
        d1_element_list.append(item[6])
        d2_element_list.append(item[7])

    def to_data_list(lst):
        out = []
        for m in lst:
            out.append(Data() if m is None else m)
        return out

    # Batch 会自动拼接 .fp 等新增属性
    return (
        Batch.from_data_list(d1_mol_list),
        Batch.from_data_list(d1_sub_list),
        Batch.from_data_list(d2_mol_list),
        Batch.from_data_list(d2_sub_list),
        Batch.from_data_list(to_data_list(d1_motif_list)),
        Batch.from_data_list(to_data_list(d2_motif_list)),
        Batch.from_data_list(to_data_list(d1_element_list)),
        Batch.from_data_list(to_data_list(d2_element_list)),
    )

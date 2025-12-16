import os
import json
import random
import numpy as np
from collections import OrderedDict
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.utils import subgraph, degree
from rdkit import Chem
from rdkit.Chem import AllChem  # >>> NEW: for fingerprint
from torch_geometric.data import Data
import rdkit.Chem.BRICS as BRICS

from utils import convert  # ensure utils.convert exists

###############################################################################
# 全局 e_map
###############################################################################
e_map = {
    'bond_type': [
        'UNSPECIFIED', 'SINGLE', 'DOUBLE', 'TRIPLE', 'QUADRUPLE', 'QUINTUPLE', 'HEXTUPLE',
        'ONEANDAHALF', 'TWOANDAHALF', 'THREEANDAHALF', 'FOURANDAHALF', 'FIVEANDAHALF',
        'AROMATIC', 'IONIC', 'HYDROGEN', 'THREECENTER', 'DATIVEONE', 'DATIVE', 'DATIVEL',
        'DATIVER', 'OTHER', 'ZERO',
    ],
    'stereo': [
        'STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


###############################################################################
# 基础函数
###############################################################################
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    """
    提取单个原子的特征(符号,氢数量,隐含价,是否芳香) + atom.GetDegree()
    """
    symbol_enc = one_of_k_encoding_unk(atom.GetSymbol(),
                                       ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                        'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd',
                                        'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
                                        'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'X'])
    h_enc = one_of_k_encoding_unk(atom.GetTotalNumHs(), [i for i in range(11)])
    valence_enc = one_of_k_encoding_unk(atom.GetImplicitValence(), [i for i in range(11)])
    is_aromatic = [atom.GetIsAromatic()]

    feat = np.array(symbol_enc + h_enc + valence_enc + is_aromatic)
    return feat, atom.GetDegree()


def calculate_shortest_path(edge_index):
    """
    计算所有节点对的最短路径距离
    """
    g = nx.DiGraph()
    g.add_edges_from(edge_index.tolist())
    s_edge_index_value = []
    all_paths = nx.all_pairs_shortest_path_length(g)
    for node_i, dist_dict in all_paths:
        for node_j, length_ij in dist_dict.items():
            s_edge_index_value.append([node_i, node_j, length_ij])
    s_edge_index_value.sort()
    return np.array(s_edge_index_value)


###############################################################################
# 1-D 指纹（ECFP2048）生成
###############################################################################
def smiles_to_fp(smiles, fp_len=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * fp_len
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_len)
    arr = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()


###############################################################################

###############################################################################
# 将SMILES转为分子图
###############################################################################
def single_smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    if c_size == 0:
        # 返回占位指纹
        return 0, [], [], [], [], [], [], 0, [0] * 2048

    features, degrees = [], []
    for atom in mol.GetAtoms():
        feat, deg_ = atom_features(atom)
        feat = feat / feat.sum() if feat.sum() != 0 else feat
        features.append(feat.tolist())
        degrees.append(deg_)

    mol_index = []
    for bond in mol.GetBonds():
        bond_type_id = e_map['bond_type'].index(str(bond.GetBondType()))
        mol_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type_id])
        mol_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), bond_type_id])

    if len(mol_index) == 0:
        return 0, [], [], [], [], [], [], 0, [0] * 2048

    mol_index = np.array(sorted(mol_index))
    mol_edge_index = mol_index[:, :2]
    mol_rel_index = mol_index[:, 2]

    s_edge_index_value = calculate_shortest_path(mol_edge_index)
    s_edge_index = s_edge_index_value[:, :2]
    s_value = s_edge_index_value[:, 2]
    s_rel = s_value.copy()
    s_rel[np.where(s_value == 1)] = mol_rel_index
    s_rel[np.where(s_value != 1)] += 23

    fp_vec = smiles_to_fp(smile)

    return (c_size, features,
            mol_edge_index.tolist(), mol_rel_index.tolist(),
            s_edge_index.tolist(), s_value.tolist(), s_rel.tolist(),
            max(degrees),
            fp_vec)


def smile_to_graph(datapath, ligands):
    sp_path = os.path.join(datapath, "mol_sp.json")
    if os.path.exists(sp_path):
        with open(sp_path, 'r') as f:
            smile_graph = json.load(f)
        max_rel_val, max_degree_val = 0, 0
        for key in smile_graph.keys():
            s_val = smile_graph[key][6]
            deg = smile_graph[key][7]
            max_rel_val = max(max_rel_val, max(s_val))
            max_degree_val = max(max_degree_val, deg)
        return smile_graph, max_rel_val, max_degree_val

    smile_graph = {}
    num_rel_mol_update = 0
    all_degree = []

    for d, smi_str in ligands.items():
        mol_obj = Chem.MolFromSmiles(smi_str)
        if not mol_obj:
            continue
        cano_smi = Chem.MolToSmiles(mol_obj)

        (c_size, feat, eidx, rel_idx, s_eidx, s_val, s_rel,
         deg_, fp_vec) = single_smile_to_graph(cano_smi)
        if c_size == 0:
            continue

        if max(s_val) > num_rel_mol_update:
            num_rel_mol_update = max(s_val)
        # 存储 fp_vec 到索引 8
        smile_graph[d] = [c_size, feat, eidx, rel_idx,
                          s_eidx, s_val, s_rel, deg_, fp_vec]
        all_degree.append(deg_)

    with open(sp_path, 'w') as f:
        json.dump(smile_graph, f)
    return smile_graph, num_rel_mol_update, max(all_degree)


###############################################################################
# 读取 interaction, network, smiles
###############################################################################
def read_interactions(path, drug_dict):
    interactions = []
    all_drug_in_ddi = []
    pos_cnt, neg_cnt = 0, 0
    positive_dict = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            drug1_id, drug2_id, rel, label = line.strip().split(" ")[:4]
            if drug1_id in drug_dict and drug2_id in drug_dict:
                all_drug_in_ddi.append(drug1_id)
                all_drug_in_ddi.append(drug2_id)
                if float(label) > 0:
                    pos_cnt += 1
                else:
                    neg_cnt += 1
                if drug1_id in positive_dict:
                    if drug2_id not in positive_dict[drug1_id]:
                        positive_dict[drug1_id].append(drug2_id)
                        interactions.append([int(drug1_id), int(drug2_id), int(rel), int(label)])
                else:
                    positive_dict[drug1_id] = [drug2_id]
                    interactions.append([int(drug1_id), int(drug2_id), int(rel), int(label)])
    print(pos_cnt, neg_cnt)
    assert pos_cnt == neg_cnt
    return np.array(interactions, dtype=int), set(all_drug_in_ddi)


def read_network(path):
    edge_index = []
    rel_index = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            head, tail, rel = line.strip().split(" ")[:3]
            edge_index.append([int(head), int(tail)])
            rel_index.append(int(rel))
    num_node = np.max(np.array(edge_index))
    num_rel = max(rel_index) + 1
    print(len(list(set(rel_index))))
    return num_node, edge_index, rel_index, num_rel


def read_smiles(path):
    print("Read", path, "!")
    out = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            drugid, smi = line.strip().split("\t")
            if drugid not in out:
                out[drugid] = smi
    return out


def motif_to_graph(datapath, ligands):
    motif_graph = {}
    motif_path = os.path.join(datapath, "motif_graph.pt")
    if os.path.exists(motif_path):
        motif_graph = torch.load(motif_path)
        return motif_graph

    print("[INFO] Generating motif graphs with structural edges …")
    for d, smi_str in ligands.items():
        mol = Chem.MolFromSmiles(smi_str)
        if not mol:
            continue

        frag_mol = Chem.FragmentOnBRICSBonds(mol)
        frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
        if not frags:
            continue

        node_feats = []
        for frag in frags:
            node_feats.append([
                frag.GetNumAtoms(),
                frag.GetNumBonds()
            ])
        x = torch.tensor(node_feats, dtype=torch.float)

        num_nodes = len(frags)
        row, col = [], []
        frag_atom_sets = [set([atom.GetAtomMapNum() for atom in frag.GetAtoms()]) for frag in frags]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if len(frag_atom_sets[i] & frag_atom_sets[j]) > 0:
                    row += [i, j]
                    col += [j, i]
        edge_index = torch.tensor([row, col], dtype=torch.long) if row else torch.zeros((2, 0), dtype=torch.long)

        motif_graph[str(d)] = Data(x=x, edge_index=edge_index)

    torch.save(motif_graph, motif_path)
    return motif_graph


def read_element_graph(path):
    data = torch.load(path)

    print(f"[DEBUG] Loaded element_graph.pt keys: {list(data.keys())}")

    if isinstance(data, dict):
        if 'graph' in data and 'element2idx' in data:
            graph = data['graph']
            element2idx = data['element2idx']
            idx2element = {i: e for e, i in element2idx.items()}

            print(f"[INFO] Loaded element graph with {graph.num_nodes} nodes and {graph.num_edges} edges")
            return graph, element2idx, idx2element
        else:
            raise KeyError("Expected keys 'graph' and 'element2idx' not found in element_graph.pt")
    else:
        raise ValueError("Unsupported format of element_graph.pt.")


def generate_element_subgraphs(ligands, element2idx, element_graph_data):
    print("[INFO] Generating element subgraphs for each drug...")
    subgraphs = {}
    edge_index = element_graph_data.edge_index
    edge_attr = element_graph_data.edge_attr
    x = element_graph_data.x
    max_index = x.size(0)

    for drug_id, smiles in ligands.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atom_indices = []

        for s in atom_symbols:
            if s not in element2idx:
                continue
            idx = element2idx[s]
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= max_index:
                continue
            atom_indices.append(idx)

        atom_indices = list(set(atom_indices))
        if len(atom_indices) == 0:
            continue

        try:
            mask = torch.tensor(atom_indices, dtype=torch.long)
            sub_edge_index, sub_edge_attr = subgraph(
                mask, edge_index, edge_attr, relabel_nodes=True
            )
            sub_x = x[mask]
        except Exception as e:
            continue

        data = Data(
            x=sub_x,
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr,
            id=torch.ones(sub_x.shape[0], dtype=torch.bool),
        )
        subgraphs[str(drug_id)] = data

    print(f"[INFO] Constructed {len(subgraphs)} element subgraphs.")
    return subgraphs


###############################################################################
# 子图提取:fixed-neighbor
###############################################################################
def generate_node_subgraphs(dataset, drug_id, network_edge_index, network_rel_index, num_rel, args):
    method = args.extractor
    edge_index = torch.from_numpy(np.array(network_edge_index).T)
    rel_index = torch.from_numpy(np.array(network_rel_index))

    row, col = edge_index
    reverse_edge_index = torch.stack((col, row), 0)
    undirected_edge_index = torch.cat((edge_index, reverse_edge_index), 1)

    paths = os.path.join("data", str(dataset), method)
    if not os.path.exists(paths):
        os.makedirs(paths, exist_ok=True)

    if method == "fixed-neighbor":
        subgraphs, max_degree, max_rel_num = fixed_neighbor_extractor(
            drug_id, undirected_edge_index, rel_index,
            paths, num_rel, fixed_num=args.fixed_num
        )
    else:
        raise ValueError("Unknown method: " + method)

    return subgraphs, max_degree, max_rel_num


def fixed_neighbor_extractor(
        drug_id_list,
        undirected_edge_index,
        rel_index,
        output_dir,
        num_rel,
        fixed_num=32
):
    json_path = os.path.join(output_dir, f"fixed_neighbor_{fixed_num}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            subgraphs = json.load(f)
            max_rel_val, max_degree_val = 0, 0
            for s in subgraphs.keys():
                if max(subgraphs[s][6]) > max_rel_val:
                    max_rel_val = max(subgraphs[s][6])
                if subgraphs[s][7] > max_degree_val:
                    max_degree_val = subgraphs[s][7]
        return subgraphs, max_degree_val, max_rel_val

    g = nx.Graph()
    g.add_edges_from(undirected_edge_index.transpose(1, 0).numpy().tolist())
    undirected_rel = torch.cat((rel_index, rel_index), 0)

    subgraphs = {}
    max_degree_list = []
    max_rel_list = []

    for d in drug_id_list:
        center = int(d)
        subsets = [center]
        neighbors = list(g.neighbors(center))
        if len(neighbors) > fixed_num:
            sampled_neighbors = random.sample(neighbors, fixed_num)
        else:
            sampled_neighbors = neighbors
        subsets.extend(sampled_neighbors)
        subsets = list(set(subsets))

        mapping_list = [False] * len(subsets)
        if center in subsets:
            mapping_list[subsets.index(center)] = True

        sub_ei, sub_ri = subgraph(
            torch.tensor(subsets, dtype=torch.long),
            undirected_edge_index,
            undirected_rel,
            relabel_nodes=True
        )
        row_sub, col_sub = sub_ei
        new_s_edge_index = sub_ei.transpose(1, 0).numpy().tolist()
        new_s_value = [1] * len(new_s_edge_index)
        new_s_rel = sub_ri.numpy().tolist()

        s_edge_index = new_s_edge_index.copy()
        s_value = new_s_value.copy()
        s_rel = new_s_rel.copy()

        edge_index_value = calculate_shortest_path(sub_ei.transpose(1, 0).numpy())
        sp_edge_index = edge_index_value[:, :2]
        sp_value = edge_index_value[:, 2]

        for i in range(len(sp_edge_index)):
            if sp_value[i] == 1:
                continue
            else:
                s_edge_index.append(sp_edge_index[i].tolist())
                s_value.append(sp_value[i])
                s_rel.append(sp_value[i] + num_rel)

        deg_val = 0
        if col_sub.numel() > 0:
            deg_val = torch.max(degree(col_sub)).item()
        max_degree_list.append(deg_val)
        rel_val = 0
        if len(s_rel) > 0:
            rel_val = int(np.max(s_rel))
        max_rel_list.append(rel_val)

        subgraphs[str(d)] = [
            subsets,
            new_s_edge_index,
            new_s_rel,
            mapping_list,
            s_edge_index,
            s_value,
            s_rel,
            deg_val
        ]

    with open(json_path, 'w') as f:
        json.dump(subgraphs, f, default=convert)
    return subgraphs, max(max_degree_list), max(max_rel_list)


def collect_neighbors_in_layer(neighbors, fixed_num=None):
    if fixed_num is None:
        return neighbors
    if neighbors.size(0) <= fixed_num:
        return neighbors
    import numpy as np
    print("networkx version:", nx.__version__)
    print("networkx location:", nx.__file__)
    print("hasattr(nx, 'Graph'):", hasattr(nx, 'Graph'))
    chosen = np.random.choice(neighbors.cpu().numpy(), size=fixed_num, replace=False)
    return torch.from_numpy(chosen).long().to(neighbors.device)


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

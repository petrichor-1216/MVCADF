# -*- coding: utf-8 -*-
import os
import torch
from rdkit import Chem
from torch_geometric.data import Data
from collections import defaultdict

def read_smiles_file(path):
    smiles_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            drug_id, smiles = parts
            smiles_dict[drug_id] = smiles
    return smiles_dict

def extract_unique_elements(smiles_dict):
    elements = set()
    for smiles in smiles_dict.values():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            elements.add(atom.GetSymbol())
    return sorted(list(elements))

def build_element_graph(element_list):
    element2idx = {e: i for i, e in enumerate(element_list)}
    idx2element = {i: e for e, i in element2idx.items()}
    edge_index = []
    edge_attr = []

    for i, e1 in enumerate(element_list):
        for j, e2 in enumerate(element_list):
            if i == j:
                continue
            edge_index.append([i, j])
            edge_attr.append([1])  # simple weight

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # use one-hot features for elements
    x = torch.eye(len(element_list), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, element2idx, idx2element

def main():
    smiles_path = "./drug_smiles.txt"
    save_path = "./element_graph.pt"

    if not os.path.exists(smiles_path):
        print(f"[ERROR] Cannot find: {smiles_path}")
        return

    smiles_dict = read_smiles_file(smiles_path)
    print(f"[INFO] Loaded {len(smiles_dict)} SMILES")

    element_list = extract_unique_elements(smiles_dict)
    print(f"[INFO] Unique elements: {len(element_list)} -> {element_list}")

    graph, element2idx, idx2element = build_element_graph(element_list)
    torch.save({
        "graph": graph,
        "element2idx": element2idx,
        "idx2element": idx2element
    }, save_path)

    print(f"[INFO] Element graph saved to: {save_path}")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

import os, argparse, random, copy, json, gc
import torch, numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from utils import *
from model.mvcadf import MVCADF
from train_eval import train, test, eval
from torch_geometric.utils import degree
from torch.utils.data.distributed import DistributedSampler
from data_process import (
    smile_to_graph, read_smiles, read_interactions,
    generate_node_subgraphs, read_network, motif_to_graph,
    read_element_graph, generate_element_subgraphs
)
import random

###############################################################################
# 参数解析
###############################################################################
def init_args(user_args=None):
    parser = argparse.ArgumentParser(description='MVCADF')
    parser.add_argument('--model_name', type=str, default='mvcadf')
    parser.add_argument('--dataset', type=str, default="drugbank")
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_episodes', type=int, default=100)
    parser.add_argument('--extractor', type=str, default="fixed-neighbor")
    parser.add_argument('--graph_fixed_num', type=int, default=1)
    parser.add_argument('--khop', type=int, default=2)
    parser.add_argument('--fixed_num', type=int, default=16)
    parser.add_argument('--num_hops', type=int, default=3)
    parser.add_argument("--d_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--max_smiles_degree", type=int, default=300)
    parser.add_argument("--max_graph_degree", type=int, default=600)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument('--sub_coeff', type=float, default=0.1)
    parser.add_argument('--mi_coeff', type=float, default=0.1)
    parser.add_argument('--s_type', type=str, default='random')
    parser.add_argument('--fp_len', type=int, default=2048)
    args = parser.parse_args()
    return args

###############################################################################
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

###############################################################################
# 其余工具函数 k_fold / split_fold 与原文件保持一致（未改动）
###############################################################################
def k_fold(data, kf, folds, y):
    test_indices = []
    train_indices = []
    if len(y):
        for _, idx in kf.split(torch.zeros(len(data)), y):
            test_indices.append(idx)
    else:
        for _, idx in kf.split(data):
            test_indices.append(idx)

    val_indices = [test_indices[i - 1] for i in range(folds)]
    for i in range(folds):
        train_mask = torch.ones(len(data), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
    return train_indices, test_indices, val_indices

def split_fold(folds, dataset, labels, scenario_type='random'):
    if scenario_type == 'random':
        skf = StratifiedKFold(folds, shuffle=True, random_state=2023)
        return k_fold(dataset, skf, folds, labels)
    raise ValueError("Unsupported split type")

###############################################################################
# 数据加载（load_data）逻辑保持原样（未改）
###############################################################################
def load_data(args):
    dataset = args.dataset
    data_path = f"dataset/{dataset}/"

    ligands = read_smiles(os.path.join(data_path, "drug_smiles.txt"))
    print("load drug smiles graphs!!")
    smile_graph, num_rel_mol_update, max_smiles_degree = smile_to_graph(data_path, ligands)

    print("load motifs from smiles!! (Motif-level rep)")
    motif_graph = motif_to_graph(data_path, ligands)

    print("load networks !!")
    num_node, network_edge_index, network_rel_index, num_rel = read_network(data_path + "networks.txt")

    print("load DDI samples!!")
    interactions_label, all_contained_drgus = read_interactions(os.path.join(data_path, "ddi.txt"), smile_graph)
    interactions = interactions_label[:, :2]
    labels = interactions_label[:, 3]

    print("generate subgraphs with method:", args.extractor)
    drug_subgraphs, max_subgraph_degree, num_rel_update = generate_node_subgraphs(
        dataset, all_contained_drgus,
        network_edge_index, network_rel_index,
        num_rel, args
    )

    print("load element graph !!")
    element_graph_data, element2idx, idx2element = read_element_graph(data_path + "element_graph.pt")
    drug_element_subgraphs = generate_element_subgraphs(ligands, element2idx, element_graph_data)

    data_sta = {
        'num_nodes': num_node + 1,
        'num_rel_mol': num_rel_mol_update + 1,
        'num_rel_graph': num_rel_update + 1,
        'num_interactions': len(interactions),
        'num_drugs_DDI': len(all_contained_drgus),
        'max_degree_graph': max_smiles_degree + 1,
        'max_degree_node': int(max_subgraph_degree) + 1
    }
    print(data_sta)

    return (interactions, labels, smile_graph, motif_graph,
            drug_subgraphs, drug_element_subgraphs, data_sta, element_graph_data)

###############################################################################
# 保存函数 save / save_results 保持不变
###############################################################################
def save(save_dir, args, train_log, test_log):
    args.device = 0
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f)
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_log, f)
    with open(os.path.join(save_dir, 'train_log.json'), 'w') as f:
        json.dump(train_log, f)

def save_results(save_dir, args, results_list):
    acc, auc, aupr, f1 = [], [], [], []
    for r in results_list:
        acc.append(r['acc']); auc.append(r['auc'])
        aupr.append(r['aupr']); f1.append(r['f1'])
    results = {
        'acc':  [float(np.mean(acc)),  float(np.std(acc))],
        'auc':  [float(np.mean(auc)),  float(np.std(auc))],
        'aupr': [float(np.mean(aupr)), float(np.std(aupr))],
        'f1':   [float(np.mean(f1)),   float(np.std(f1))]
    }
    d = vars(args).copy(); d.update(results)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, args.extractor + '_all_results.json'), 'a+') as f:
        json.dump(d, f)

###############################################################################
# init_model: 传入 fp_len
###############################################################################
def init_model(args, stats, element_graph_data):
    if args.model_name != 'mvcadf':
        raise ValueError("Unknown model name")

    model = MVCADF(
        max_layer=args.layer,
        num_features_drug=67,
        num_nodes=stats['num_nodes'],
        num_relations_mol=stats['num_rel_mol'],
        num_relations_graph=stats['num_rel_graph'],
        output_dim=args.d_dim,
        max_degree_graph=stats['max_degree_graph'],
        max_degree_node=stats['max_degree_node'],
        sub_coeff=args.sub_coeff,
        mi_coeff=args.mi_coeff,
        dropout=args.dropout,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        element_feature_dim=element_graph_data.x.shape[1],
        fp_len=args.fp_len                     # >>> NEW: 传递指纹长度
    )
    optim = torch.optim.Adam(model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)
    return model, optim

###############################################################################
def main():
    args = init_args()
    interactions, labels, smile_g, motif_g, node_g, elem_subg, stats, elem_graph = load_data(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setup_seed(42)
    train_ids, test_ids, val_ids = split_fold(args.folds, interactions, labels, args.s_type)

    results = []
    for fold, (tr_idx, te_idx, va_idx) in enumerate(zip(train_ids, test_ids, val_ids)):
        print(f"============================{fold+1}/{args.folds}==================================")
        feat_dim = elem_graph.x.shape[1]

        ds_kwargs = dict(sub_graph=node_g, smile_graph=smile_g,
                         motif_graph=motif_g, element_graph=elem_subg,
                         element_feature_dim=feat_dim, fp_len=args.fp_len)  # >>> NEW

        train_data = DTADataset(x=interactions[tr_idx], y=labels[tr_idx], **ds_kwargs)
        test_data  = DTADataset(x=interactions[te_idx], y=labels[te_idx], **ds_kwargs)
        eval_data  = DTADataset(x=interactions[va_idx], y=labels[va_idx], **ds_kwargs)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                   shuffle=True, collate_fn=collate)
        test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=args.batch_size,
                                                   shuffle=False, collate_fn=collate)
        eval_loader  = torch.utils.data.DataLoader(eval_data,  batch_size=args.batch_size,
                                                   shuffle=False, collate_fn=collate)

        model, optimizer = init_model(args, stats, elem_graph)
        model.to(device)
        if hasattr(model, "reset_parameters"):            # >>> NEW safeguard
            model.reset_parameters()

        best_auc, best_state, patience = 0., None, 0
        train_log = {k: [] for k in ['train_acc','train_auc','train_aupr','train_loss',
                                     'eval_acc','eval_auc','eval_aupr','eval_loss']}

        for epoch in range(args.model_episodes):
            loop = tqdm(train_loader, ncols=80,
                        desc=f'Epoch[{epoch}/{args.model_episodes}]')
            tr_acc,tr_f1,tr_auc,tr_aupr,tr_loss = train(train_loader, model, optimizer)
            ev_acc,ev_f1,ev_auc,ev_aupr,ev_loss = eval(eval_loader, model)
            print(f"train_auc:{tr_auc:} train_aupr:{tr_aupr:} "
                  f"eval_auc:{ev_auc:} eval_aupr:{ev_aupr:}")

            for k,v in zip(['train_acc','train_auc','train_aupr','train_loss',
                            'eval_acc','eval_auc','eval_aupr','eval_loss'],
                           [tr_acc,tr_auc,tr_aupr,tr_loss,ev_acc,ev_auc,ev_aupr,ev_loss]):
                train_log[k].append(v)

            if ev_auc > best_auc:
                best_auc, best_state, patience = ev_auc, copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1
                if patience > 10:
                    print("early stop!"); break

        model.load_state_dict(best_state)
        model.to(device)
        test_log = test(test_loader, model)

        save_dir = os.path.join('./best_save/', args.model_name, args.dataset,
                                args.extractor, f"fold_{fold}",
                                "{:.5f}".format(test_log['auc']))
        os.makedirs(save_dir, exist_ok=True)
        model.save(save_dir)
        save(save_dir, args, train_log, test_log)
        print(f"save to {save_dir}")
        results.append(test_log)

    save_results(os.path.join('./best_save/', args.model_name, args.dataset),
                 args, results)

if __name__ == "__main__":
    main()

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve,
    accuracy_score, auc
)
from tqdm import tqdm

# -----------------------------------------------------------
def contains_nan(tensor: torch.Tensor):
    return bool(torch.isnan(tensor).any())

# -----------------------------------------------------------
def train(loader, model, optimizer):
    model.train()
    total_loss = 0.0

    prob_all, label_all = [], []

    loop = tqdm(loader, ncols=80)
    for batch_idx, data in enumerate(loop):
        dm1, dg1, dm2, dg2, mt1, mt2, el1, el2 = data

        if torch.cuda.is_available():
            dm1, dg1, dm2, dg2 = dm1.cuda(), dg1.cuda(), dm2.cuda(), dg2.cuda()
            mt1, mt2, el1, el2 = mt1.cuda(), mt2.cuda(), el1.cuda(), el2.cuda()

        optimizer.zero_grad(set_to_none=True)

        predicts, loss = model(dm1, dg1, dm2, dg2, mt1, mt2, el1, el2)

        if contains_nan(loss) or contains_nan(predicts):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # >>> FIX: 下面三行必须在循环内部
        prob_all.append(predicts.detach())
        label_all.append(dm1.y.detach())
        total_loss += loss.item() * dm1.num_graphs           # <<< FIX

    # 循环结束后做聚合 ---------------------------------
    if len(prob_all) == 0:
        return 0, 0, 0, 0, float('inf')

    prob_all = torch.cat(prob_all).cpu().numpy()
    label_all = torch.cat(label_all).cpu().numpy()

    train_acc, train_f1, train_auc, train_aupr = get_score(label_all, prob_all)
    return train_acc, train_f1, train_auc, train_aupr, total_loss / len(loader.dataset)

# -----------------------------------------------------------
@torch.no_grad()
def eval(loader, model):
    model.eval()
    total_loss = 0.0
    prob_all, label_all = [], []

    for idx, data in enumerate(loader):
        dm1, dg1, dm2, dg2, mt1, mt2, el1, el2 = data
        if torch.cuda.is_available():
            dm1, dg1, dm2, dg2 = dm1.cuda(), dg1.cuda(), dm2.cuda(), dg2.cuda()
            mt1, mt2, el1, el2 = mt1.cuda(), mt2.cuda(), el1.cuda(), el2.cuda()

        predicts, loss = model(dm1, dg1, dm2, dg2, mt1, mt2, el1, el2)
        if contains_nan(loss) or contains_nan(predicts):
            continue

        prob_all.append(predicts); label_all.append(dm1.y)
        total_loss += loss.item() * dm1.num_graphs

    if len(prob_all) == 0:
        return 0, 0, 0, 0, float('inf')

    prob_all = torch.cat(prob_all).cpu().numpy()
    label_all = torch.cat(label_all).cpu().numpy()

    ev_acc, ev_f1, ev_auc, ev_aupr = get_score(label_all, prob_all)
    return ev_acc, ev_f1, ev_auc, ev_aupr, total_loss / len(loader.dataset)

# -----------------------------------------------------------
@torch.no_grad()
def test(loader, model):
    model.eval()
    total_loss = 0.0
    prob_all, label_all = [], []

    for idx, data in enumerate(loader):
        dm1, dg1, dm2, dg2, mt1, mt2, el1, el2 = data
        if torch.cuda.is_available():
            dm1, dg1, dm2, dg2 = dm1.cuda(), dg1.cuda(), dm2.cuda(), dg2.cuda()
            mt1, mt2, el1, el2 = mt1.cuda(), mt2.cuda(), el1.cuda(), el2.cuda()

        predicts, loss = model(dm1, dg1, dm2, dg2, mt1, mt2, el1, el2)
        if contains_nan(loss) or contains_nan(predicts):
            continue

        prob_all.append(predicts); label_all.append(dm1.y)
        total_loss += loss.item() * dm1.num_graphs

    if len(prob_all) == 0:
        return {"acc":0, "f1":0, "auc":0, "aupr":0, "loss":float('inf')}

    prob_all = torch.cat(prob_all).cpu().numpy()
    label_all = torch.cat(label_all).cpu().numpy()

    te_acc, te_f1, te_auc, te_aupr = get_score(label_all, prob_all)
    return {"acc":te_acc, "f1":te_f1, "auc":te_auc,
            "aupr":te_aupr, "loss":total_loss / len(loader.dataset)}

# -----------------------------------------------------------
def get_score(label_all, prob_all):
    pred_label = (prob_all >= 0.5).astype(int)
    acc = accuracy_score(label_all, pred_label)
    f1  = f1_score(label_all, pred_label)
    auroc = roc_auc_score(label_all, prob_all)
    p, r, _ = precision_recall_curve(label_all, prob_all)
    auprc = auc(r, p)
    return acc, f1, auroc, auprc

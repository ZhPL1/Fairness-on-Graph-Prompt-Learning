import torchmetrics
import torch
from tqdm import tqdm
import numpy as np

# def calculate_fairness_metrics(pred, true_labels, batch_indices, idx_test, group_0, group_1, num_classes, device):
def calculate_fairness_metrics(pred, labels, idx_test, group_0, group_1, device):

    y_hat = pred[idx_test].cpu().numpy()
    y = labels[idx_test].cpu().numpy()

    idx_s0 = group_0
    idx_s1 = group_1

    idx_s0_y1 = np.bitwise_and(idx_s0,y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,y==1)
    
    parity = abs(sum(y_hat[idx_s0]) / sum(idx_s0) - sum(y_hat[idx_s1]) / sum(idx_s1))
    equality = abs(sum(y_hat[idx_s0_y1]) / sum(idx_s0_y1) - sum(y_hat[idx_s1_y1]) / sum(idx_s1_y1))

    return parity, equality
    # list0 = group_0.tolist()
    # list1 = idx_test.tolist()
    # common_elements = set(list0).intersection(set(list1))
    # group_0_indices = torch.tensor(list(common_elements)).to(device)
    # list0 = group_1.tolist()
    # list1 = idx_test.tolist()
    # common_elements = set(list0).intersection(set(list1))
    # group_1_indices = torch.tensor(list(common_elements)).to(device)

    # for y in range(num_classes):
    #     if len(group_0_indices) == 0:
    #         pred_counts_group_0[y] = 0
    #     else:
    #         pred_counts_group_0[y] = (pred[group_0_indices] == torch.tensor(y)).sum().to(device)
    #         pred_counts_group_0[y] = pred_counts_group_0[y] / len(group_0_indices)
        
    #     if len(group_1_in_idx_test) == 0:
    #         pred_counts_group_1[y] = 0
    #     else:
    #         pred_counts_group_1[y] = (pred[group_1_in_idx_test] == torch.tensor(y)).sum().to(device)
    #         pred_counts_group_1[y] = pred_counts_group_1[y] / len(group_1_in_idx_test)

    #     count_0 = (labels.to(device)[group_0_indices]  == torch.tensor(y).to(device)).sum().item()
    #     count_1 = (labels.to(device)[group_1_in_idx_test]  == torch.tensor(y).to(device)).sum().item()

    #     if count_0 == 0:
    #         cond_pred_counts_group_0[y] = 0
    #     else:
    #         cond_pred_counts_group_0[y] = ((pred[group_0_indices] == torch.tensor(y).to(device)) & (labels[group_0_indices] == torch.tensor(y).to(device))).sum().item()
    #         cond_pred_counts_group_0[y] = cond_pred_counts_group_0[y] / count_0
        
    #     if count_1 == 0:
    #         cond_pred_counts_group_1[y] = 0
    #     else:
    #         cond_pred_counts_group_1[y] = ((pred[group_1_in_idx_test] == torch.tensor(y).to(device)) & (labels[group_1_in_idx_test] == torch.tensor(y).to(device))).sum().item()
    #         cond_pred_counts_group_1[y] = cond_pred_counts_group_1[y] / count_1

    # P_pred_y_group_0 = pred_counts_group_0 
    # P_pred_y_group_1 = pred_counts_group_1 

    # P_pred_y_cond_true_y_group_0 = cond_pred_counts_group_0 
    # P_pred_y_cond_true_y_group_1 = cond_pred_counts_group_1

    # fairness_metric_1 = torch.abs(P_pred_y_group_0 - P_pred_y_group_1).sum().item() / num_classes
    # fairness_metric_2 = torch.abs(P_pred_y_cond_true_y_group_0 - P_pred_y_cond_true_y_group_1).sum().item() / num_classes


def GNNNodeEva(data, idx_test, gnn, answering, num_class, device, group_0=None, group_1=None, if_fair=False):
    gnn.eval()
    # loader = data
    idx_test = idx_test.to(device)
    data = data.to(device)
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()

    out = gnn(data.x, data.edge_index, batch=None)
    out = answering(out)
    pred = out.argmax(dim=1)

    print("pred:{}".format(pred))
    print("data.y:{}".format(data.y))

    if if_fair:
        # group_0 = group_0.to(device)
        # group_1 = group_1.to(device)
        dp, eo = calculate_fairness_metrics(pred, data.y, idx_test, group_0, group_1, device)
        
    acc = accuracy(pred[idx_test], data.y[idx_test])
    f1 = macro_f1(pred[idx_test], data.y[idx_test])
    roc = auroc(out[idx_test], data.y[idx_test])
    prc = auprc(out[idx_test], data.y[idx_test])

    if if_fair:
        return acc.item(), f1.item(), roc.item(), prc.item(), dp, eo
    else:
        return acc.item(), f1.item(), roc.item(), prc.item()

def GNNGraphEva(loader, gnn, answering, num_class, device):
    gnn.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()
    if answering:
        answering.eval()
    with torch.no_grad(): 
        for batch_id, batch in enumerate(loader): 
            batch = batch.to(device) 
            out = gnn(batch.x, batch.edge_index, batch.batch)
            if answering:
                out = answering(out)  
            pred = out.argmax(dim=1)  
            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            roc = auroc(out, batch.y)
            prc = auprc(out, batch.y)
            if len(loader) > 20:
                print("Batch {}/{} Acc: {:.4f} | Macro-F1: {:.4f}| AUROC: {:.4f}| AUPRC: {:.4f}".format(batch_id,len(loader), acc.item(), ma_f1.item(),roc.item(), prc.item()))

            # print(acc)
    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    roc = auroc.compute()
    prc = auprc.compute()
       
    return acc.item(), ma_f1.item(), roc.item(), prc.item()
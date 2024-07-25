import torchmetrics
import torch
from tqdm import tqdm
import numpy as np
import math


# def calculate_fairness_metrics(pred, true_labels, batch_indices, idx_test, group_0, group_1, num_classes, device):

#     global_indices = idx_test[batch_indices]
#     pred_batch = pred
#     true_labels_batch = true_labels

#     list0 = group_0.tolist()
#     list1 = global_indices.tolist()
#     common_elements = set(list0).intersection(set(list1))
#     batch_group_0_indices = torch.tensor(list(common_elements)).to(device)

#     list0 = group_1.tolist()
#     list1 = global_indices.tolist()
#     common_elements = set(list0).intersection(set(list1))
#     batch_group_1_indices = torch.tensor(list(common_elements)).to(device)

#     group_0_in_idx_test = torch.nonzero(global_indices.unsqueeze(0) == batch_group_0_indices.unsqueeze(1))[:,1]
#     group_1_in_idx_test = torch.nonzero(global_indices.unsqueeze(0) == batch_group_1_indices.unsqueeze(1))[:,1]

#     pred_counts_group_0 = torch.zeros(num_classes).to(device)
#     pred_counts_group_1 = torch.zeros(num_classes).to(device)

#     cond_pred_counts_group_0 = torch.zeros(num_classes).to(device)
#     cond_pred_counts_group_1 = torch.zeros(num_classes).to(device)

#     for y in range(num_classes):
#         if len(group_0_in_idx_test) == 0:
#             pred_counts_group_0[y] = 0
#         else:
#             pred_counts_group_0[y] = (pred_batch[group_0_in_idx_test] == torch.tensor(y).to(device)).sum().to(device)
#             pred_counts_group_0[y] = pred_counts_group_0[y] / len(group_0_in_idx_test)
        
        
#         if len(group_1_in_idx_test) == 0:
#             pred_counts_group_1[y] = 0
#         else:
#             pred_counts_group_1[y] = (pred_batch[group_1_in_idx_test] == torch.tensor(y).to(device)).sum().to(device)
#             pred_counts_group_1[y] = pred_counts_group_1[y] / len(group_1_in_idx_test)

#         # print("pred_batch:{}".format(pred_batch[group_0_in_idx_test] == torch.tensor(y).to(device)))
#         # print("pred_batch:{}".format(pred_counts_group_0[y]))

#         count_0 = (true_labels_batch[group_0_in_idx_test] == torch.tensor(y).to(device)).sum().item()
#         count_1 = (true_labels_batch[group_1_in_idx_test] == torch.tensor(y).to(device)).sum().item()

#         if count_0 == 0:
#             cond_pred_counts_group_0[y] = 0
#         else:
#             cond_pred_counts_group_0[y] = ((pred_batch[group_0_in_idx_test] == torch.tensor(y).to(device)) & (true_labels_batch[group_0_in_idx_test] == torch.tensor(y).to(device))).sum().item()
#             cond_pred_counts_group_0[y] = cond_pred_counts_group_0[y] / count_0
        
#         if count_1 == 0:
#             cond_pred_counts_group_1[y] = 0
#         else:
#             cond_pred_counts_group_1[y] = ((pred_batch[group_1_in_idx_test] == torch.tensor(y).to(device)) & (true_labels_batch[group_1_in_idx_test] == torch.tensor(y).to(device))).sum().item()
#             cond_pred_counts_group_1[y] = cond_pred_counts_group_1[y] / count_1

#     P_pred_y_group_0 = pred_counts_group_0 
#     P_pred_y_group_1 = pred_counts_group_1 

#     P_pred_y_cond_true_y_group_0 = cond_pred_counts_group_0 
#     P_pred_y_cond_true_y_group_1 = cond_pred_counts_group_1

#     fairness_metric_1 = torch.abs(P_pred_y_group_0 - P_pred_y_group_1).sum().item() / num_classes
#     fairness_metric_2 = torch.abs(P_pred_y_cond_true_y_group_0 - P_pred_y_cond_true_y_group_1).sum().item() / num_classes


#     return fairness_metric_1, fairness_metric_2

def calculate_fairness_metrics(pred, labels, idx_test, group_0, group_1, device):

    y_hat = pred.cpu().numpy()
    y = labels.cpu().numpy()

    idx_s0 = group_0
    idx_s1 = group_1

    idx_s0_y1 = np.bitwise_and(idx_s0,y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,y==1)
    
    parity = abs(sum(y_hat[idx_s0]) / sum(idx_s0) - sum(y_hat[idx_s1]) / sum(idx_s1))
    equality = abs(sum(y_hat[idx_s0_y1]) / sum(idx_s0_y1) - sum(y_hat[idx_s1_y1]) / sum(idx_s1_y1))

    print("y_hat:{}".format(y_hat))
    print("y:{}".format(y))

    return parity, equality


def AllInOneEva(loader, prompt, gnn, answering, num_class, device, idx_test=None, group_0=None, group_1=None, if_fair=False):
    prompt.eval()
    answering.eval()


    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()

    pred_all = torch.Tensor().to(device)
    label_all = torch.Tensor().to(device)

    with torch.no_grad():
        for batch_id, (batch, _) in enumerate(loader):
            batch = batch.to(device)
            prompted_graph = prompt(batch)
            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            pre = answering(graph_emb)
            pred = pre.argmax(dim=1)

            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            roc = auroc(pre, batch.y)
            prc = auprc(pre, batch.y)
            if len(loader) > 20:
                print("Batch {}/{} Acc: {:.4f} | Macro-F1: {:.4f}| AUROC: {:.4f}| AUPRC: {:.4f}".format(batch_id,len(loader), acc.item(), ma_f1.item(),roc.item(), prc.item()))

            pred_all = torch.cat((pred_all, pred), dim=0)
            label_all = torch.cat((label_all, batch.y), dim=0)
    # print("pred_all:{}".format(pred_all))
    # print("label_all:{}".format(label_all))
    # print("len_pred_all:{}".format(len(pred_all)))
    # print("len_label_all:{}".format(len(label_all)))

    if if_fair:
        dp, eo = calculate_fairness_metrics(pred_all, label_all, idx_test, group_0, group_1, device)

    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    roc = auroc.compute()
    prc = auprc.compute()

    if if_fair:
        return acc.item(), ma_f1.item(), roc.item(), prc.item(), dp, eo
    else:
        return acc.item(), ma_f1.item(), roc.item(), prc.item()
   


def AllInOneEvaWithoutAnswer(loader, prompt, gnn, num_class, device):
        prompt.eval()
        accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
        macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
        accuracy.reset()
        macro_f1.reset()
        for batch_id, test_batch in enumerate(loader):
            test_batch = test_batch.to(device)
            emb0 = gnn(test_batch.x, test_batch.edge_index, test_batch.batch)
            pg_batch = prompt.token_view()
            pg_batch = pg_batch.to(device)
            pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
            dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
            pre = torch.softmax(dot, dim=1)

            y = test_batch.y
            pre_cla = torch.argmax(pre, dim=1)

            acc = accuracy(pre_cla, y)
            ma_f1 = macro_f1(pre_cla, y)

        acc = accuracy.compute()
        ma_f1 = macro_f1.compute()
        return acc
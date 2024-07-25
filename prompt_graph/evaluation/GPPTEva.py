import torchmetrics
import torch
from tqdm import tqdm
import numpy as np

def calculate_fairness_metrics(pred, labels, idx_test, group_0, group_1, device):

    y_hat = pred[idx_test].cpu().numpy()
    y = labels[idx_test].cpu().numpy()

    idx_s0 = group_0
    idx_s1 = group_1

    idx_s0_y1 = np.bitwise_and(idx_s0,y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,y==1)
    
    parity = abs(sum(y_hat[idx_s0]) / sum(idx_s0) - sum(y_hat[idx_s1]) / sum(idx_s1))
    equality = abs(sum(y_hat[idx_s0_y1]) / sum(idx_s0_y1) - sum(y_hat[idx_s1_y1]) / sum(idx_s1_y1))

    print("y_hat[idx_s0]:{}".format(y_hat[idx_s0]))
    print("y_hat[idx_s1]:{}".format(y_hat[idx_s1]))

    return parity, equality


def GPPTEva(data, idx_test, gnn, prompt, num_class, device, group_0=None, group_1=None, if_fair=False):
    # gnn.eval()
    prompt.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()

    node_embedding = gnn(data.x.to(device), data.edge_index.to(device))
    out = prompt(node_embedding.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=1).to(device)  
    
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

def GPPTGraphEva(loader, gnn, prompt, num_class, device):
    # batch must be 1
    prompt.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)
    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()
    with torch.no_grad(): 
        for batch_id, batch in enumerate(loader): 
            batch=batch.to(device)              
            node_embedding = gnn(batch.x,batch.edge_index)
            out = prompt(node_embedding, batch.edge_index)
            

            # 找到每个预测中概率最大的索引（类别）
            predicted_classes = out.argmax(dim=1)

            # # 统计每个类别获得的票数
            votes = predicted_classes.bincount(minlength=out.shape[1])

            # # 找出票数最多的类别
            pred = votes.argmax()
            pred = pred.unsqueeze(dim=-1)
            average_out = torch.nn.functional.softmax(votes.float(), dim=0).unsqueeze(dim=0)

            # correct += int((pred == batch.y).sum())  
            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            roc = auroc(average_out, batch.y)
            prc = auprc(average_out, batch.y)
            if len(loader) > 20:
                print("Batch {}/{} Acc: {:.4f} | Macro-F1: {:.4f}| AUROC: {:.4f}| AUPRC: {:.4f}".format(batch_id,len(loader), acc.item(), ma_f1.item(),roc.item(), prc.item()))

    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    roc = auroc.compute()
    prc = auprc.compute()
       
    return acc.item(), ma_f1.item(), roc.item(), prc.item()
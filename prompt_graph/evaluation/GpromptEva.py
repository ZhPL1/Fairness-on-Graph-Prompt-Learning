import torch.nn.functional as F
import torchmetrics
import torch
from tqdm import tqdm
import numpy as np

def calculate_fairness_metrics(pred, labels, idx_test, group_0, group_1, device):

    y_hat = pred.cpu().numpy()
    y = labels.cpu().numpy()

    idx_s0 = group_0
    idx_s1 = group_1

    idx_s0_y1 = np.bitwise_and(idx_s0,y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,y==1)
    
    parity = abs(sum(y_hat[idx_s0]) / sum(idx_s0) - sum(y_hat[idx_s1]) / sum(idx_s1))
    equality = abs(sum(y_hat[idx_s0_y1]) / sum(idx_s0_y1) - sum(y_hat[idx_s1_y1]) / sum(idx_s1_y1))

    print("y_hat[idx_s0]:{}".format(y_hat[idx_s0]))
    print("y_hat[idx_s1]:{}".format(y_hat[idx_s1]))
    print("y[idx_s0]:{}".format(y[idx_s0]))
    print("y[idx_s1]:{}".format(y[idx_s1]))

    return parity, equality


def GpromptEva(loader, gnn, prompt, center_embedding, num_class, device, idx_test=None, group_0=None, group_1=None, if_fair=False):
    prompt.eval()
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
        for batch_id, (batch, batch_indices) in enumerate(loader): 
            batch = batch.to(device) 
            out = gnn(batch.x, batch.edge_index, batch.batch, prompt, 'Gprompt')
            similarity_matrix = F.cosine_similarity(out.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1)
            pred = similarity_matrix.argmax(dim=1)

            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            roc = auroc(similarity_matrix, batch.y)
            prc = auprc(similarity_matrix, batch.y)
            if len(loader) > 20:
                print("Batch {}/{} Acc: {:.4f} | Macro-F1: {:.4f}| AUROC: {:.4f}| AUPRC: {:.4f}".format(batch_id,len(loader), acc.item(), ma_f1.item(),roc.item(), prc.item()))

            pred_all = torch.cat((pred_all, pred), dim=0)
            label_all = torch.cat((label_all, batch.y), dim=0)

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
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer
from prompt_graph.utils import Gprompt_link_loss
from prompt_graph.utils import edge_index_to_sparse_matrix, prepare_structured_data
from prompt_graph.data import load4link_prediction_single_graph,load4link_prediction_multi_graph
import time
from .base import PreTrain
import os

class Edgepred_Gprompt(PreTrain):
    def __init__(self, graph_list, input_dim, out_dim, *args, **kwargs):    
        super().__init__(graph_list=graph_list, input_dim=input_dim, out_dim=out_dim, *args, **kwargs)
        self.dataloader = self.generate_loader_data()
        self.initialize_gnn(self.input_dim, self.hid_dim) 
        self.graph_pred_linear = torch.nn.Linear(self.hid_dim, self.output_dim).to(self.device)  

    def generate_loader_data(self):    
        if self.dataset_name in ['PubMed', 'CiteSeer', 'Cora', 'Actor', 'Computers', 'Photo','ogbn-arxiv', 'Flickr','Actor', 'Texas', 'Wisconsin', 'nba', 'pokec_z', 'pokec_n', 'credit', 'german']:            
            self.data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_single_graph(self.dataset_name)
            self.adj = edge_index_to_sparse_matrix(self.data.edge_index, self.data.x.shape[0]).to(self.device)
            data = prepare_structured_data(self.data)
            if self.dataset_name in['ogbn-arxiv', 'Flickr']:
                return DataLoader(TensorDataset(data), batch_size = 1024, shuffle=True)
            else:
                return DataLoader(TensorDataset(data), batch_size=64, shuffle=True)
        
        elif self.dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'ogbg-ppa', 'DD']:
            self.data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_multi_graph(self.dataset_name)          
            self.adj = edge_index_to_sparse_matrix(self.data.edge_index, self.data.x.shape[0]).to(self.device)
            data = prepare_structured_data(self.data)

            if self.dataset_name in  ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa', 'DD']:
                from torch_geometric import loader
                self.batch_dataloader = loader.DataLoader(self.data.to_data_list(),batch_size=256,shuffle=False)
                return DataLoader(TensorDataset(data), batch_size=5120000, shuffle=True)
            else:
                return DataLoader(TensorDataset(data), batch_size=64, shuffle=True)
    
    def pretrain_one_epoch(self):
        accum_loss, total_step = 0, 0
        device = self.device
        self.gnn.train()
        for step, batch in enumerate(self.dataloader): 
            self.optimizer.zero_grad()

            batch = batch[0]
            batch = batch.to(device)

            # 如果graph datasets经过Batch图太大了，那就分开操作
            if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa', 'DD']:

                for batch_id, batch_graph in enumerate(self.batch_dataloader):
                    batch_graph.to(device)
                    if(batch_id==0):
                        out = self.gnn(batch_graph.x, batch_graph.edge_index)
                    else:
                        out = torch.concatenate([out, self.gnn(batch_graph.x, batch_graph.edge_index)],dim=0)
            else:
                out = self.gnn(self.data.x.to(device), self.data.edge_index.to(device))
               
            all_node_emb = self.graph_pred_linear(out)

            # TODO: GraphPrompt customized node embedding computation
            all_node_emb = torch.sparse.mm(self.adj,all_node_emb)
    
            node_emb = all_node_emb[batch[:, 0]]
            pos_emb, neg_emb = all_node_emb[batch[:, 1]], all_node_emb[batch[:, 2]]

            loss = Gprompt_link_loss(node_emb, pos_emb, neg_emb)

            loss.backward()
            self.optimizer.step()

            accum_loss += float(loss.detach().cpu().item())
            total_step += 1
            
            # print('第{}次反向传播过程'.format(step))

        return accum_loss / total_step
 
    def pretrain(self):
        num_epoch = self.epochs
        train_loss_min = 1000000
        patience = 10
        cnt_wait = 0

        for epoch in range(1, num_epoch + 1):
            st_time = time.time()
            train_loss = self.pretrain_one_epoch()

            print(f"Edgepred_Gprompt [Pretrain] Epoch {epoch}/{num_epoch} | Train Loss {train_loss:.5f} | "
                  f"Cost Time {time.time() - st_time:.3}s")
            
            if train_loss_min > train_loss:
                train_loss_min = train_loss
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait == patience:
                    print('-' * 100)
                    print('Early stopping at '+str(epoch) +' eopch!')
                    break
            print(cnt_wait)

        folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(self.gnn.state_dict(),
                    "./Experiment/pre_trained_model/{}/{}.{}.{}.pth".format(self.dataset_name, 'Edgepred_Gprompt', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
        
        print("+++model saved ! {}/{}.{}.{}.pth".format(self.dataset_name, 'Edgepred_Gprompt', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
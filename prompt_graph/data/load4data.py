import torch
import pickle as pk
from random import shuffle
import random
import torch_sparse
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr, WebKB, Actor
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data,Batch
from torch_geometric.utils import negative_sampling
import os
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
import numpy as np
import pandas as pd
import scipy.sparse as sp

# def node_sample_and_save(data, k, folder, num_classes):
#     # 获取标签
#     labels = data.y.to('cpu')
#     label_idx = np.where(labels>=0)[0]
#     num_test = int(0.25*len(label_idx))
#     if num_test < 320:
#         num_test = 100
#     test_idx = torch.randperm(data.num_nodes)[:num_test]
#     test_labels = labels[test_idx]

#     remaining_idx = torch.randperm(data.num_nodes)[num_test:]
#     remaining_labels = labels[remaining_idx]
    
#     train_idx = torch.cat([remaining_idx[remaining_labels == i][:k] for i in range(num_classes)])
#     shuffled_indices = torch.randperm(train_idx.size(0))
#     train_idx = train_idx[shuffled_indices]
#     train_labels = labels[train_idx]

#     train_idx = torch.cat([train_idx[train_labels == i][:k] for i in range(num_classes)])
#     shuffled_indices = torch.randperm(train_idx.size(0))
#     train_idx = train_idx[shuffled_indices]
#     train_labels = labels[train_idx]
#     torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
#     torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
#     torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
#     torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))

def node_sample_and_save(data, k, folder, num_classes):
    labels = data.y.to('cpu')

    if len(labels) == 2708: # cora
        # 随机选择90%的数据作为测试集
        num_test = int(0.9 * data.num_nodes)
        if num_test < 1000:
            num_test = int(0.7 * data.num_nodes)
        test_idx = torch.randperm(data.num_nodes)[:num_test]
        test_labels = labels[test_idx]
        
        remaining_idx = torch.randperm(data.num_nodes)[num_test:]
        remaining_labels = labels[remaining_idx]


    elif len(labels) == 30000 : # credit 
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]  
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        idx_test = np.append(
            label_idx_0[int(0.75 * len(label_idx_0)) :],
            label_idx_1[int(0.75 * len(label_idx_1)) :],
        )
        test_idx = torch.tensor(idx_test)
        test_labels = labels[test_idx]
        remaining_idx = torch.tensor(np.append(
            label_idx_0[0:3000],
            label_idx_1[0:3000])
        )
        remaining_labels = labels[remaining_idx]

    
    elif len(labels) == 18876: # bail
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]  
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        idx_test = np.append(
            label_idx_0[int(0.75 * len(label_idx_0)) :],
            label_idx_1[int(0.75 * len(label_idx_1)) :],
        )
        test_idx = torch.tensor(idx_test)
        test_labels = labels[test_idx]
        remaining_idx = torch.tensor(np.append(
            label_idx_0[0:100],
            label_idx_1[0:100])
        )
        remaining_labels = labels[remaining_idx]


    else: # pokec_n, pokec_z, nba
        label_idx = np.where(labels>=0)[0]
        num_test = int(0.25*len(label_idx))
        if num_test < 200: #nba     
            num_test = 100
        test_idx = torch.tensor(label_idx[-num_test:])
        test_labels = labels[test_idx]

        if num_test == 100: # nba
            remaining_idx = torch.tensor(label_idx[: min(int(0.5 * len(label_idx)), num_test)])
        else: # pokec_n, pokec_z
            remaining_idx = torch.tensor(label_idx[: min(int(0.5 * len(label_idx)), 500)]) 
            remaining_labels = labels[remaining_idx]
    
    train_idx = torch.cat([remaining_idx[remaining_labels == i][:k] for i in range(num_classes)])
    shuffled_indices = torch.randperm(train_idx.size(0))
    train_idx = train_idx[shuffled_indices]
    train_labels = labels[train_idx]
    


    torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))


def graph_sample_and_save(dataset, k, folder, num_classes):

    num_graphs = len(dataset)
    num_test = int(0.25 * num_graphs)

    labels = torch.tensor([graph.y.item() for graph in dataset])

    # 随机选择测试集的图索引
    all_indices = torch.randperm(num_graphs)
    test_indices = all_indices[:num_test]
    torch.save(test_indices, os.path.join(folder, 'test_idx.pt'))
    test_labels = labels[test_indices]
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))

    remaining_indices = all_indices[num_test:]

    # 从剩下的10%的图中为训练集选择每个类别的k个样本
    train_indices = []
    for i in range(num_classes):
        # 选出该类别的所有图
        class_indices = [idx for idx in remaining_indices if labels[idx].item() == i]
        # 如果选出的图少于k个，就取所有该类的图
        selected_indices = class_indices[:k] 
        train_indices.extend(selected_indices)

    # 随机打乱训练集的图索引
    train_indices = torch.tensor(train_indices)
    shuffled_indices = torch.randperm(train_indices.size(0))
    train_indices = train_indices[shuffled_indices]
    torch.save(train_indices, os.path.join(folder, 'train_idx.pt'))
    train_labels = labels[train_indices]
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))

def node_degree_as_features(data_list):
    from torch_geometric.utils import degree
    for data in data_list:
        # 计算所有节点的度数，这将返回一个张量
        deg = degree(data.edge_index[0], dtype=torch.long)

        # 将度数张量变形为[nodes, 1]以便与其他特征拼接
        deg = deg.view(-1, 1).float()
        
        # 如果原始数据没有节点特征，可以直接使用度数作为特征
        if data.x is None:
            data.x = deg
        else:
            # 将度数特征拼接到现有的节点特征上
            data.x = torch.cat([data.x, deg], dim=1)

def load4graph(dataset_name, shot_num= 10, num_parts=None, pretrained=False):
    r"""A plain old python object modeling a batch of graphs as one big
        (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
        base class, all its methods can also be used here.
        In addition, single graphs can be reconstructed via the assignment vector
        :obj:`batch`, which maps each node to its respective graph identifier.
        """

    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'DD']:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        if dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY']:
            graph_list = [g for g in graph_list]
            node_degree_as_features(graph_list)
            input_dim = graph_list[0].x.size(1)        

        # # 分类并选择每个类别的图
        # class_datasets = {}
        # for data in dataset:
        #     label = data.y.item()
        #     if label not in class_datasets:
        #         class_datasets[label] = []
        #     class_datasets[label].append(data)

        # train_data = []
        # remaining_data = []
        # for label, data_list in class_datasets.items():
        #     train_data.extend(data_list[:shot_num])
        #     random.shuffle(train_data)
        #     remaining_data.extend(data_list[shot_num:])

        # # 将剩余的数据 1：9 划分为测试集和验证集
        # random.shuffle(remaining_data)
        # val_dataset_size = len(remaining_data) // 9
        # val_dataset = remaining_data[:val_dataset_size]
        # test_dataset = remaining_data[val_dataset_size:]
        

        if(pretrained==True):
            return input_dim, out_dim, graph_list
        else:
            return input_dim, out_dim, dataset
        
    if dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = PygGraphPropPredDataset(name = dataset_name, root='./dataset')
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        graph_list = [g for g in graph_list]
        node_degree_as_features(graph_list)
        input_dim = graph_list[0].x.size(1)

        for g in graph_list:
            g.y = g.y.squeeze(0)

        if(pretrained==True):
            return input_dim, out_dim, graph_list
        else:
            return input_dim, out_dim, dataset        


    if  dataset_name in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]
        num_parts=200

        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
        
        dataset = list(ClusterData(data=data, num_parts=num_parts))
        graph_list = dataset
        # 这里的图没有标签

        return input_dim, out_dim, data
    
    if dataset_name in ['Wisconsin', 'Texas']:
        dataset = WebKB(root='data/WebKB', name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]
        num_parts=200
        
        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
        
        # dataset = list(ClusterData(data=data, num_parts=num_parts))
        # graph_list = dataset
        # 这里的图没有标签

        return input_dim, out_dim, data

    if dataset_name in ['Actor']:
        dataset = Actor(root='data/Actor', transform=NormalizeFeatures())
        data = dataset[0]

        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
        return input_dim, out_dim, data


    if dataset_name in ['nba']:
        idx_features_labels = pd.read_csv(os.path.join('data/NBA', 'nba.csv'))
        header = list(idx_features_labels.columns)
        header.remove("user_id")
        header.remove("country")
        header.remove("SALARY")
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels["SALARY"].values
        labels = torch.tensor(labels, dtype=torch.long)
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join('data/NBA', 'nba_relationship.txt'), dtype=int)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        input_dim = features.shape[1]
        sens = idx_features_labels["country"].values
        edges = torch.Tensor(edges).transpose(0, 1)
        edge_index = to_undirected(edges).int()
        x = torch.Tensor(features.todense())
        labels[labels>1] = 1
        sens[sens>0] = 1
        out_dim = torch.unique(labels>=0).size(0)
        data = Data(x=x, edge_index=edge_index, y=labels)
        return input_dim, out_dim, data


    if dataset_name in ['pokec_z']:
        idx_features_labels = pd.read_csv(os.path.join('data/pokec', 'region_job.csv'))
        header = list(idx_features_labels.columns)
        header.remove("user_id")
        header.remove("region")
        header.remove("I_am_working_in_field")
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels["I_am_working_in_field"].values
        labels = torch.tensor(labels, dtype=torch.long)
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join('data/pokec', 'region_job_relationship.txt'), dtype=int)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        input_dim = features.shape[1]
        sens = idx_features_labels["region"].values
        edges = torch.Tensor(edges).transpose(0, 1)
        edge_index = to_undirected(edges).int()
        x = torch.Tensor(features.todense()) 
        labels[labels>1] = 1
        sens[sens>0] = 1
        out_dim = torch.unique(labels>=0).size(0)
        data = Data(x=x, edge_index=edge_index, y=labels)
        return input_dim, out_dim, data


    if dataset_name in ['pokec_n']:
        idx_features_labels = pd.read_csv(os.path.join('data/pokec', 'region_job_2.csv'))
        header = list(idx_features_labels.columns)
        header.remove("user_id")
        header.remove("region")
        header.remove("I_am_working_in_field")
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels["I_am_working_in_field"].values
        labels = torch.tensor(labels, dtype=torch.long)
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join('data/pokec', 'region_job_2_relationship.txt')).astype("int")
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        input_dim = features.shape[1]
        sens = idx_features_labels["region"].values
        edges = torch.Tensor(edges).transpose(0, 1)
        edge_index = to_undirected(edges).int()
        x = torch.Tensor(features.todense()) 
        labels[labels>1] = 1
        sens[sens>0] = 1
        out_dim = torch.unique(labels>=0).size(0)
        data = Data(x=x, edge_index=edge_index, y=labels)

        return input_dim, out_dim, data
    
    if dataset_name in ['credit']:
        idx_features_labels = pd.read_csv(os.path.join('data/credit', 'credit.csv'))
        header = list(idx_features_labels.columns)
        header.remove('NoDefaultNextMonth')
        header.remove("Single")
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels['NoDefaultNextMonth'].values
        labels = torch.tensor(labels, dtype=torch.long)
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join('data/credit', 'credit_edges.txt')).astype("int")
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        input_dim = features.shape[1]
        sens = idx_features_labels["Age"].values
        edges = torch.Tensor(edges).transpose(0, 1)
        edge_index = to_undirected(edges).to(torch.int64)
        x = torch.Tensor(features.todense()) 
        out_dim = len(torch.unique(labels))
        data = Data(x=x, edge_index=edge_index, y=labels)

        return input_dim, out_dim, data

    if dataset_name in ['creditA']:
        idx_features_labels = pd.read_csv(os.path.join('data/credit', 'creditA.csv'))
        header = list(idx_features_labels.columns)
        header.remove('NoDefaultNextMonth')
        header.remove("Single")
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels['NoDefaultNextMonth'].values
        labels = torch.tensor(labels, dtype=torch.long)
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join('data/credit', 'creditA_edges.txt')).astype("int")
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        input_dim = features.shape[1]
        sens = idx_features_labels["Age"].values
        edges = torch.Tensor(edges).transpose(0, 1)
        edge_index = to_undirected(edges).to(torch.int64)
        x = torch.Tensor(features.todense()) 
        out_dim = len(torch.unique(labels))
        data = Data(x=x, edge_index=edge_index, y=labels)

        return input_dim, out_dim, data

    if dataset_name in ['bail']:
        idx_features_labels = pd.read_csv(os.path.join('data/bail', 'bail.csv'))
        header = list(idx_features_labels.columns)
        header.remove("RECID")
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels["RECID"].values
        labels = torch.LongTensor(labels)
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join('data/bail', 'bail_edges.txt')).astype("int")
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        input_dim = features.shape[1]
        sens = idx_features_labels["WHITE"].values
        edges = torch.Tensor(edges).transpose(0, 1)
        edge_index = to_undirected(edges).to(torch.int64)
        x = torch.Tensor(features.todense()) 
        out_dim = len(torch.unique(labels))
        labels = torch.LongTensor(labels)
        data = Data(x=x, edge_index=edge_index, y=labels)
    
    if dataset_name in ['german']:
        idx_features_labels = pd.read_csv(os.path.join('data/bail', 'german.csv'))
        header = list(idx_features_labels.columns)
        header.remove("GoodCustomer")
        header.remove("OtherLoansAtStore")
        header.remove("PurposeOfLoan")

        idx_features_labels["Gender"][idx_features_labels["Gender"] == "Female"] = 1
        idx_features_labels["Gender"][idx_features_labels["Gender"] == "Male"] = 0

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels["GoodCustomer"].values
        labels[labels == -1] = 0
        labels = torch.LongTensor(labels)
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join('data/bail', 'german_edges.txt')).astype("int")
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        input_dim = features.shape[1]
        sens = idx_features_labels["Gender"].values
        edges = torch.Tensor(edges).transpose(0, 1)
        edge_index = to_undirected(edges).to(torch.int64)
        x = torch.Tensor(features.todense()) 
        out_dim = len(torch.unique(labels))
        labels = torch.LongTensor(labels)
        data = Data(x=x, edge_index=edge_index, y=labels)

        return input_dim, out_dim, data



    
def load4node(dataname):
    print(dataname)
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='data/Planetoid', name=dataname, transform=NormalizeFeatures())
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='data/amazon', name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Reddit':
        dataset = Reddit(root='data/Reddit')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'WikiCS':
        dataset = WikiCS(root='data/WikiCS')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Flickr':
        dataset = Flickr(root='data/Flickr')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['Wisconsin', 'Texas']:
        dataset = WebKB(root='data/'+dataname, name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Actor':
        dataset = Actor(root='data/Actor')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./dataset')
        data = dataset[0]
        input_dim = data.x.shape[1]
        out_dim = dataset.num_classes
    elif dataname == 'nba':
        idx_features_labels = pd.read_csv(os.path.join('data/NBA', 'nba.csv'))
        header = list(idx_features_labels.columns)
        header.remove("user_id")
        header.remove("country")
        header.remove("SALARY")
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels["SALARY"].values
        labels = torch.tensor(labels, dtype=torch.long)
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join('data/NBA', 'nba_relationship.txt'), dtype=int)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        input_dim = features.shape[1]
        sens = idx_features_labels["country"].values
        edges = torch.Tensor(edges).transpose(0, 1)
        edge_index = to_undirected(edges).to(torch.int64)
        x = torch.Tensor(features.todense())
        labels[labels>1] = 1
        sens[sens>0] = 1 
        out_dim = torch.unique(labels>=0).size(0)
        data = Data(x=x, edge_index=edge_index, y=labels)

    elif dataname == 'pokec_z':
        idx_features_labels = pd.read_csv(os.path.join('data/pokec', 'region_job.csv'))
        header = list(idx_features_labels.columns)
        header.remove("user_id")
        header.remove("region")
        header.remove("I_am_working_in_field")
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels["I_am_working_in_field"].values
        labels = torch.tensor(labels, dtype=torch.long)
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join('data/pokec', 'region_job_relationship.txt'), dtype=int)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        input_dim = features.shape[1]
        sens = idx_features_labels["region"].values
        edges = torch.Tensor(edges).transpose(0, 1)
        edge_index = to_undirected(edges).to(torch.int64)
        x = torch.Tensor(features.todense()) 
        labels[labels>1] = 1
        sens[sens>0] = 1 
        out_dim = torch.unique(labels>=0).size(0)
        data = Data(x=x, edge_index=edge_index, y=labels)

    elif dataname == 'pokec_n':
        idx_features_labels = pd.read_csv(os.path.join('data/pokec', 'region_job_2.csv'))
        header = list(idx_features_labels.columns)
        header.remove("user_id")
        header.remove("region")
        header.remove("I_am_working_in_field")
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels["I_am_working_in_field"].values
        labels = torch.tensor(labels, dtype=torch.long)
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join('data/pokec', 'region_job_2_relationship.txt'), dtype=int)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        input_dim = features.shape[1]
        sens = idx_features_labels["region"].values
        edges = torch.Tensor(edges).transpose(0, 1)
        edge_index = to_undirected(edges).to(torch.int64)
        x = torch.Tensor(features.todense()) 
        labels[labels>1] = 1
        sens[sens>0] = 1 
        out_dim = torch.unique(labels>=0).size(0)
        data = Data(x=x, edge_index=edge_index, y=labels)

    elif dataname == 'credit':
        idx_features_labels = pd.read_csv(os.path.join('data/credit', 'credit.csv'))
        header = list(idx_features_labels.columns)
        header.remove('NoDefaultNextMonth')
        header.remove("Single")
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels['NoDefaultNextMonth'].values
        labels = torch.tensor(labels, dtype=torch.long)
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join('data/credit', 'credit_edges.txt')).astype("int")
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        input_dim = features.shape[1]
        sens = idx_features_labels["Age"].values
        edges = torch.Tensor(edges).transpose(0, 1)
        edge_index = to_undirected(edges).to(torch.int64)
        x = torch.Tensor(features.todense()) 
        out_dim = len(torch.unique(labels))
        data = Data(x=x, edge_index=edge_index, y=labels)

    elif dataname == 'creditA':
        idx_features_labels = pd.read_csv(os.path.join('data/credit', 'creditA.csv'))
        header = list(idx_features_labels.columns)
        header.remove('NoDefaultNextMonth')
        header.remove("Single")
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels['NoDefaultNextMonth'].values
        labels = torch.tensor(labels, dtype=torch.long)
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join('data/credit', 'creditA_edges.txt')).astype("int")
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        input_dim = features.shape[1]
        sens = idx_features_labels["Age"].values
        edges = torch.Tensor(edges).transpose(0, 1)
        edge_index = to_undirected(edges).to(torch.int64)
        x = torch.Tensor(features.todense()) 
        out_dim = len(torch.unique(labels))
        data = Data(x=x, edge_index=edge_index, y=labels)

    elif dataname == 'bail':
        idx_features_labels = pd.read_csv(os.path.join('data/bail', 'bail.csv'))
        header = list(idx_features_labels.columns)
        header.remove("RECID")
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels["RECID"].values
        labels = torch.LongTensor(labels)
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join('data/bail', 'bail_edges.txt')).astype("int")
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        input_dim = features.shape[1]
        sens = idx_features_labels["WHITE"].values
        edges = torch.Tensor(edges).transpose(0, 1)
        edge_index = to_undirected(edges).to(torch.int64)
        x = torch.Tensor(features.todense()) 
        out_dim = len(torch.unique(labels))
        labels = torch.LongTensor(labels)
        data = Data(x=x, edge_index=edge_index, y=labels)

    elif dataname == 'german':
        idx_features_labels = pd.read_csv(os.path.join('data/bail', 'german.csv'))
        header = list(idx_features_labels.columns)
        header.remove("GoodCustomer")
        header.remove("OtherLoansAtStore")
        header.remove("PurposeOfLoan")

        idx_features_labels["Gender"][idx_features_labels["Gender"] == "Female"] = 1
        idx_features_labels["Gender"][idx_features_labels["Gender"] == "Male"] = 0

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels["GoodCustomer"].values
        labels = torch.LongTensor(labels)
        labels[labels == -1] = 0
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join('data/bail', 'german_edges.txt')).astype("int")
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
        input_dim = features.shape[1]
        sens = idx_features_labels["Gender"].values
        edges = torch.Tensor(edges).transpose(0, 1)
        edge_index = to_undirected(edges).to(torch.int64)
        x = torch.Tensor(features.todense()) 
        out_dim = len(torch.unique(labels))
        labels = torch.LongTensor(labels)
        data = Data(x=x, edge_index=edge_index, y=labels)

    if dataname in ['nba', 'pokec_z', 'pokec_n', 'credit', 'bail']:
        return data, input_dim, out_dim, sens
    else:
        return data, input_dim, out_dim, data


    # print()
    # print(f'Dataset: {dataset}:')
    # print('======================')
    # print(f'Number of graphs: {len(dataset)}')
    # print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')

    # data = dataset[0]  # Get the first graph object.

    # print()
    # print(data)
    # print('===========================================================================================================')

    # # Gather some statistics about the graph.
    # print(f'Number of nodes: {data.num_nodes}')
    # print(f'Number of edges: {data.num_edges}')
    # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    # print(f'Has self-loops: {data.has_self_loops()}')
    # print(f'Is undirected: {data.is_undirected()}')

    #  # 根据 shot_num 更新训练掩码
    # class_counts = {}  # 统计每个类别的节点数
    # for label in data.y:
    #     label = label.item()
    #     class_counts[label] = class_counts.get(label, 0) + 1

    
    # train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # # val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    
    # for label in data.y.unique():
    #     label_indices = (data.y == label).nonzero(as_tuple=False).view(-1)

    #     # if len(label_indices) < 3 * shot_num:
    #     #     raise ValueError(f"类别 {label.item()} 的样本数不足以分配到训练集、测试集和验证集。")

    #     label_indices = label_indices[torch.randperm(len(label_indices))]
    #     train_indices = label_indices[:shot_num]
    #     train_mask[train_indices] = True       
    #     remaining_indices = label_indices[100:]
    #     # split_point = int(len(remaining_indices) * 0.1)  # 验证集占剩余的10%
        
    #     # val_indices = remaining_indices[:split_point]
    #     test_indices = remaining_indices

    #     # val_mask[val_indices] = True
    #     test_mask[test_indices] = True

    # data.train_mask = train_mask
    # data.test_mask = test_mask
    # # data.val_mask = val_mask


def load4link_prediction_single_graph(dataname, num_per_samples=1):
    data, input_dim, output_dim, _ = load4node(dataname)

    
    r"""Perform negative sampling to generate negative neighbor samples"""
    if data.is_directed():
        row, col = data.edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = data.edge_index
        
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges * num_per_samples,
    )

    edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)

    return data, edge_label, edge_index, input_dim, output_dim

def load4link_prediction_multi_graph(dataset_name, num_per_samples=1):
    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'DD']:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)

    if dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = PygGraphPropPredDataset(name = dataset_name, root='./dataset')
    
    input_dim = dataset.num_features
    output_dim = 2 # link prediction的输出维度应该是2，0代表无边，1代表右边

    if dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY']:
        dataset = [g for g in dataset]
        node_degree_as_features(dataset)
        input_dim = dataset[0].x.size(1)

    elif dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = [g for g in dataset]
        node_degree_as_features(dataset)
        input_dim = dataset[0].x.size(1)
        for g in dataset:
            g.y = g.y.squeeze(1)

    data = Batch.from_data_list(dataset)
    
    r"""Perform negative sampling to generate negative neighbor samples"""
    if data.is_directed():
        row, col = data.edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = data.edge_index
        
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges * num_per_samples,
    )

    edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)
    
    return data, edge_label, edge_index, input_dim, output_dim

# 未完待续，需要重写一个能够对large-scale图分类数据集的划分代码，避免node-level和edge-level的预训练算法或prompt方法显存溢出的问题
def load4link_prediction_multi_large_scale_graph(dataset_name, num_per_samples=1):
    if dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = PygGraphPropPredDataset(name = dataset_name, root='./dataset')
    
    input_dim = dataset.num_features
    output_dim = 2 # link prediction的输出维度应该是2，0代表无边，1代表右边

    dataset = [g for g in dataset]
    node_degree_as_features(dataset)
    input_dim = dataset[0].x.size(1)
    for g in dataset:
        g.y = g.y.squeeze(1)

    batch_graph_num = 20000
    split_num = int(len(dataset)/batch_graph_num)
    data_list = []
    edge_label_list = []
    edge_index_list = []
    for i in range(split_num+1):
        if(i==0):
            data = Batch.from_data_list(dataset[0:batch_graph_num])
        elif(i<=split_num):
            data = Batch.from_data_list(dataset[i*batch_graph_num:(i+1)*batch_graph_num])
        elif len(dataset)>((i-1)*batch_graph_num):
            data = Batch.from_data_list(dataset[i*batch_graph_num:(i+1)*batch_graph_num])
        

        data_list.append(data)
        
        r"""Perform negative sampling to generate negative neighbor samples"""
        if data.is_directed():
            row, col = data.edge_index
            row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
            edge_index = torch.stack([row, col], dim=0)
        else:
            edge_index = data.edge_index
            
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.num_edges * num_per_samples,
        )

        edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
        edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)
    
    return data, edge_label, edge_index, input_dim, output_dim

# used in pre_train.py
def NodePretrain(data, num_parts=20, split_method='Random Walk'):

    if(split_method=='Cluster'):
        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        
        graph_list = list(ClusterData(data=data, num_parts=num_parts))
    elif(split_method=='Random Walk'):
        from torch_cluster import random_walk
        split_ratio = 0.1
        walk_length = 30
        all_random_node_list = torch.randperm(data.num_nodes)
        selected_node_num_for_random_walk = int(split_ratio * data.num_nodes)
        random_node_list = all_random_node_list[:selected_node_num_for_random_walk]
        walk_list = random_walk(data.edge_index[0].to(torch.int64), data.edge_index[1].to(torch.int64), random_node_list, walk_length=walk_length)

        graph_list = [] 
        skip_num = 0        
        for walk in walk_list:   
            subgraph_nodes = torch.unique(walk)
            if(len(subgraph_nodes)<5):
                skip_num+=1
                continue
            subgraph_data = data.subgraph(subgraph_nodes)

            graph_list.append(subgraph_data)

        print(f"Total {len(graph_list)} random walk subgraphs with nodes more than 5, and there are {skip_num} skipped subgraphs with nodes less than 5.")

    else:
        print('None split method!')
        exit()
    
    # return list(data)
    return graph_list



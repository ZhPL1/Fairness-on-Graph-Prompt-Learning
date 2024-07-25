from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import warnings
import torch
warnings.filterwarnings("ignore")

class GraphDataset(Dataset):
    def __init__(self, graphs):
        """
        初始化 GraphDataset
        :param graphs: 包含图对象的列表
        """
        self.graphs = graphs
        # self._indices = None

    def __len__(self):
        """
        返回数据集的大小
        :return: 数据集的大小
        """
        return len(self.graphs)

    def __getitem__(self, idx):
        """
        获取索引为 idx 的图
        :param idx: 索引
        :return: 图对象
        """

        graph = self.graphs[idx]
        # graph.idx = torch.tensor([idx])
        # 可以在这里进行图数据的预处理或特征提取
        # 例如，如果每个图对象都有节点特征和边特征，可以返回它们
        # return {'node_features': graph.node_features, 'edge_index': graph.edge_index}
        return graph, idx
        # return self.graphs[idx]

    # def get(self, idx):
    #     """
    #     Get the graph object at index idx.
    #     :param idx: Index.
    #     :return: Graph object.
    #     """
    #     graph = self.graphs[idx]
    #     return graph

    # def __getitem__(self, idx):
    #     """
    #     Get the graph object at index idx.
    #     :param idx: Index.
    #     :return: Graph object.
    #     """
    #     if isinstance(idx, int):
    #         return self.get(idx)
    #     elif isinstance(idx, slice):
    #         # Handle slicing
    #         return [self.get(i) for i in range(*idx.indices(len(self)))]
    #     elif isinstance(idx, list) or isinstance(idx, np.ndarray) or isinstance(idx, torch.Tensor):
    #         # Handle list, numpy array, or tensor of indices
    #         return [self.get(i) for i in idx]
    #     else:
    #         raise TypeError(f"Unsupported index type: {type(idx)}")
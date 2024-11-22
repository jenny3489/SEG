"""
Raw image -> Superpixel segmentation -> graph
"""
import numpy as np
import torch
import cv2 as cv
from torch_scatter import scatter
from torch_geometric.data import Data
import copy
from torch import nn


# Getting adjacent relationship among nodes
def get_edge_index(segment):
    if isinstance(segment, torch.Tensor):
        segment = segment.numpy()#将张量转换为 NumPy 数组
    # 扩张
    img = segment.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)#卷积核3×3，全是1
    expansion = cv.dilate(img, kernel)#dilate：膨胀，扩大图像中高亮区域，通常是白色区域或1值区域
    mask = segment == expansion#布尔掩码，掩码标识了在膨胀前后未变化的像素位置，没有被膨胀的在掩码中被标True
    mask = np.invert(mask)#反转掩码，对变化的元素掩码True，关注点转向变化区域，即原始图像的边界和细节变化之处。
    # 构图
    h, w = segment.shape
    edge_index = set()
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    indices = list(zip(*np.nonzero(mask)))#取掩mask中所有值为True（或非零）的像素坐标
    for x, y in indices: #对于每一个非零像素坐标 (x, y)，
        for dx, dy in directions:#遍历其周围八个相邻像素
            adj_x, adj_y = x + dx, y + dy
            if -1 < adj_x < h and -1 < adj_y < w:# 检查这些相邻像素是否位于图像范围内。如果相邻像素位于范围内，
                source, target = segment[x, y], segment[adj_x, adj_y]
                if source != target: # 且与当前像素的值不同
                    edge_index.add((source, target))
                    edge_index.add((target, source)) # 则将这对像素的值作为一条边添加到 edge_index 集合中，并且为了保证图的无向性，同时添加反向的边
    return torch.tensor(list(edge_index), dtype=torch.long).T, edge_index#将edge_index集合转换为一个PyTorch张量，类型为torch.long，并且转置（.T）

# Getting node features
def get_node(x, segment, mode='mean'):
    assert x.ndim == 3 and segment.ndim == 2
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(segment, np.ndarray):#浮点型张量
        segment = torch.from_numpy(segment).to(torch.long)#长整型
    c = x.shape[2]#将特征矩阵x从形状(H, W, C)重塑为(H*W, C)，这样做是为了将每个像素的特征向量平铺开来，便于后续按区域聚合。
    x = x.reshape((-1, c))#-1：自动计算该维度的大小，使总元素数量保持不变，调用reshape((-1, c))后，它的形状会变成(height * width, channels)
    mask = segment.flatten()
    nodes = scatter(x, mask, dim=0, reduce=mode)#使用scatter函数按照segment中的标签对特征向量进行聚合，reduce=mode指定了聚合的方式，常见的模式有'mean'（平均）、'sum'（求和）等
    return nodes.to(torch.float32)


# Constructing graphs by shifting
def get_grid_adj(grid):
    edge_index = list()
    # 上偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:-1] = grid[1:]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 下偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[1:] = grid[:-1]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 左偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:, :-1] = grid[:, 1:]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 右偏移
    a = np.full_like(grid, -1, dtype=np.int32)
    a[:, 1:] = grid[:, :-1]
    adj = np.stack([grid, a], axis=-1)
    mask = adj != -1
    mask = np.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    return edge_index


# Getting graph list
def get_graph_list(data, seg):

    graph_node_feature = []
    graph_edge_index = []
    for i in np.unique(seg):
        # 获取节点特征
        graph_node_feature.append(data[seg == i])
        # 获取邻接信息
        x, y = np.nonzero(seg == i)
        n = len(x)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        grid = np.full((x_max - x_min + 1, y_max - y_min + 1), -1, dtype=np.int32)
        x_hat, y_hat = x - x_min, y - y_min
        grid[x_hat, y_hat] = np.arange(n)
        graph_edge_index.append(get_grid_adj(grid))
    graph_list = []
    # 数据变换
    for node, edge_index in zip(graph_node_feature, graph_edge_index):
        node = torch.tensor(node, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        graph_list.append(Data(node, edge_index=edge_index))
    return graph_list


def split(graph_list, gt, mask):
    indices = np.nonzero(gt)
    ans = []
    number = mask[indices]
    gt = gt[indices]
    for i, n in enumerate(number):
        graph = copy.deepcopy(graph_list[n])
        graph.y = torch.tensor([gt[i]], dtype=torch.long)
        ans.append(graph)
    return ans


def summary(net: nn.Module):
    single_dotted_line = '-' * 40
    double_dotted_line = '=' * 40
    star_line = '*' * 40
    content = []
    def backward(m: nn.Module, chain: list):
        children = m.children()
        params = 0
        chain.append(m._get_name())
        try:
            child = next(children)
            params += backward(child, chain)
            for child in children:
                params += backward(child, chain)
            # print('*' * 40)
            # print('{:>25}{:>15,}'.format('->'.join(chain), params))
            # print('*' * 40)
            if content[-1] is not star_line:
                content.append(star_line)
            content.append('{:>25}{:>15,}'.format('->'.join(chain), params))
            content.append(star_line)
        except:
            for p in m.parameters():
                if p.requires_grad:
                    params += p.numel()
            # print('{:>25}{:>15,}'.format(chain[-1], params))
            content.append('{:>25}{:>15,}'.format(chain[-1], params))
        chain.pop()
        return params
    # print('-' * 40)
    # print('{:>25}{:>15}'.format('Layer(type)', 'Param'))
    # print('=' * 40)
    content.append(single_dotted_line)
    content.append('{:>25}{:>15}'.format('Layer(type)', 'Param'))
    content.append(double_dotted_line)
    params = backward(net, [])
    # print('=' * 40)
    # print('-' * 40)
    content.pop()
    content.append(single_dotted_line)
    print('\n'.join(content))
    return params



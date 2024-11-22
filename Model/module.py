from torch_geometric import nn as gnn
import torch
from torch import nn
from torch.nn import functional as F


# Internal graph convolution
class SubGcn(nn.Module):
    def __init__(self, c_in, hidden_size, nc):
        super().__init__()
        self.gcn = gnn.SGConv(c_in, hidden_size, K=3)
        # 1
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hidden_size // 2, nc)
        )

    def forward(self, graph):
        h = F.relu(self.gcn(graph.x, graph.edge_index))
        h_avg = gnn.global_mean_pool(h, graph.batch)
        logits = self.classifier(h_avg)
        return logits
class SubGcnFeature(nn.Module):
    def __init__(self, c_in, hidden_size, gat_heads=1, dropout=0.5):
        super().__init__()
        self.gcn1 = gnn.SGConv(c_in, hidden_size, K=3)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.gcn2 = gnn.GraphConv(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.gat = gnn.GATConv(hidden_size, hidden_size, heads=gat_heads)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        h1 = F.relu(self.gcn1(graph.x, graph.edge_index))
        h1 = self.bn1(h1)
        h2 = F.relu(self.gcn2(h1, graph.edge_index))
        h2 = self.bn2(h2)
        h3 = F.relu(self.gat(h2, graph.edge_index))
        h3 = self.bn3(h3)
        h = h1 + h3  # 残差连接
        h = self.dropout(h)  # dropout
        h_avg = gnn.global_mean_pool(h, graph.batch)
        return h_avg

class GraphNet(nn.Module):
    def __init__(self, c_in, hidden_size, nc, dropout=0.5, num_heads=8):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size)
        self.bn_1 = gnn.BatchNorm(hidden_size)
        self.gcn_2 = gnn.GraphConv(hidden_size, hidden_size)
        self.bn_2 = gnn.BatchNorm(hidden_size)
        self.gat_1 = gnn.GATConv(hidden_size, hidden_size // 8, heads=8)
        self.bn_3 = gnn.BatchNorm(hidden_size)
        print(nc)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, nc)

        )

    def forward(self, graph):
        x = self.bn_0(graph.x)
        # print(x)
        h = self.gcn_1(x,graph.edge_index)
        # print(h)
        h = F.relu(h)
        h = self.bn_1(h)
        h = F.relu(self.gcn_2(h, graph.edge_index))
        h = self.bn_2(h)
        h = F.relu(self.gat_1(h, graph.edge_index))
        h = self.bn_3(h)

        h = self.ffn(h) + x  # 加入残差连接

        h = h.unsqueeze(0)  # 增加序列维度，以适应多头注意力机制的输入格式 (S, N, E)
        attn_output, _ = self.attention(h, h, h)
        attn_output = attn_output.squeeze(0)  # 去掉序列维度

        logits = self.classifier(attn_output + x)
        return logits


# External graph convolution feature module
class GraphNetFeature(nn.Module):
    def __init__(self, c_in, hidden_size):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size)
        self.bn_1 = gnn.BatchNorm(hidden_size)
        self.gcn_2 = gnn.GCNConv(hidden_size, hidden_size)
        self.bn_2 = gnn.BatchNorm(hidden_size)

    def forward(self, graph):
        x_normalization = self.bn_0(graph.x)
        # x_normalization = graph.x
        h = self.bn_1(F.relu(self.gcn_1(x_normalization, graph.edge_index)))
        h = self.bn_2(F.relu(self.gcn_2(h, graph.edge_index)))
        return x_normalization + h
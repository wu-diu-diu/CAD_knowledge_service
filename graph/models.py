from layers import GraphConvolution, GATLayer, GraphSAGELayer
import torch.nn as nn
import torch.nn.functional as F
import torch

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)  ## 定义第一层图卷积层，输入特征维度为nfeat，输出特征维度为nhid
        self.gc2 = GraphConvolution(nhid, nclass)  ## 定义第二层图卷积层，输入特征维度为nhid，输出特征维度为nclass
        self.dropout = dropout  ## 定义dropout的概率
    
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  ## 第一层图卷积层的前向传播，输入特征矩阵x和邻接矩阵adj，输出经过ReLU激活函数处理的特征矩阵
        x = F.dropout(x, self.dropout, training=self.training)  ## 对第一层的输出进行dropout操作，使用定义的dropout概率，并且只在训练模式下应用dropout
        x = self.gc2(x, adj)  ## 第二层图卷积层的前向传播，输入经过dropout处理的特征矩阵和邻接矩阵，输出最终的特征矩阵
        return F.log_softmax(x, dim=1)  ## 对第二层的输出进行log_softmax操作，得到每个节点所属类别的概率分布


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout  ## 定义dropout的概率
        ## 定义多头注意力机制的图卷积层，使用ModuleList存储多个GATLayer，每个GATLayer的输入特征维度为nfeat，输出特征维度为nhid，dropout概率为dropout，负斜率为alpha，concat参数设置为True
        self.attentions = nn.ModuleList([GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        ## 定义输出层的图卷积层，输入特征维度为nhid乘以nheads（因为多头注意力机制会将每个头的输出特征拼接起来），输出特征维度为nclass，dropout概率为dropout，负斜率为alpha，concat参数设置为False
        self.out_att = GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False) 
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  ## 对输入特征矩阵x进行dropout操作，使用定义的dropout概率，并且只在训练模式下应用dropout
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  ## 对每个注意力头进行前向传播，将每个头的输出特征矩阵拼接起来，得到一个新的特征矩阵
        x = F.dropout(x, self.dropout, training=self.training)  ## 对拼接后的特征矩阵进行dropout操作，使用定义的dropout概率，并且只在训练模式下应用dropout
        x = self.out_att(x, adj)  ## 输出层的前向传播，输入经过dropout处理的特征矩阵和邻接矩阵，输出最终的特征矩阵
        return F.log_softmax(x, dim=1)  ## 对输出层的输出进行log_softmax操作，得到每个节点所属类别的概率分布


class GraphSAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSAGE, self).__init__()
        ## 定义两层GraphSAGELayer，第一层输入特征维度为nfeat，输出特征维度为nhid；第二层输入特征维度为nhid，输出特征维度为nclass。dropout参数定义了dropout的概率。
        self.sage1 = GraphSAGELayer(nfeat, nhid)
        self.sage2 = GraphSAGELayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, first_hop_neighbors, second_hop_neighbors):
        ## x 的形状是 (N, nfeat)，表示整张图所有节点的输入特征。
        ## first_hop_neighbors 的形状是 (N, 2)，表示每个目标节点采样得到的 2 个一阶邻居索引。
        ## second_hop_neighbors 的形状是 (N, 2, 3)，表示每个目标节点的每个一阶邻居再采样得到的 3 个二阶邻居索引。
        num_nodes, first_fanout = first_hop_neighbors.shape  ## 取出节点总数和一阶邻居采样数。
        second_fanout = second_hop_neighbors.size(2)  ## 取出二阶邻居采样数。

        root_first_hop_features = x[first_hop_neighbors]  ## 根据一阶邻居索引取出每个目标节点的一阶邻居特征，形状是 (N, 2, nfeat)。
        root_hidden = self.sage1(x, root_first_hop_features)  ## 第一条分支：把目标节点自身特征和它的一阶邻居特征做一次 GraphSAGE 聚合。
        root_hidden = F.relu(root_hidden)  ## 对目标节点的隐藏表示做 ReLU 激活。
        root_hidden = F.dropout(root_hidden, self.dropout, training=self.training)  ## 在训练阶段对目标节点隐藏表示做 dropout。

        first_hop_self_features = x[first_hop_neighbors].reshape(num_nodes * first_fanout, -1)  ## 把所有一阶邻居节点拉平成一个批次，作为下一步聚合的“中心节点”。
        first_hop_neighbor_features = x[second_hop_neighbors].reshape(num_nodes * first_fanout, second_fanout, -1)  ## 取出每个一阶邻居对应采样到的二阶邻居特征。
        first_hop_hidden = self.sage1(first_hop_self_features, first_hop_neighbor_features)  ## 第二条分支：先为每个一阶邻居聚合它自己的二阶邻居，得到一阶邻居的隐藏表示。
        first_hop_hidden = F.relu(first_hop_hidden)  ## 对一阶邻居隐藏表示做 ReLU 激活。
        first_hop_hidden = F.dropout(first_hop_hidden, self.dropout, training=self.training)  ## 在训练阶段对一阶邻居隐藏表示做 dropout。
        first_hop_hidden = first_hop_hidden.view(num_nodes, first_fanout, -1)  ## 把一阶邻居隐藏表示重新整理回 (N, 2, nhid) 的结构。

        output = self.sage2(root_hidden, first_hop_hidden)  ## 用第二层 GraphSAGE 把目标节点隐藏表示和一阶邻居隐藏表示再次聚合，得到分类前输出。
        return F.log_softmax(output, dim=1)  ## 对输出做 log_softmax，方便后续直接使用 NLLLoss 训练。

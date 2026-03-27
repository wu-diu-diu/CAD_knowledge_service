import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_features,)))
        else:
            self.register_parameter('bias', None)  ## 如果不使用偏置项，则注册一个名为'bias'的参数，并将其设置为None
        self.reset_parameters()  ## 调用reset_parameters方法初始化权重参数
    
    def reset_parameters(self):
        stdv = 1. / self.weight.size(1)  ## 计算权重参数的标准差，使用权重矩阵的列数进行计算
        self.weight.data.uniform_(-stdv, stdv)  ## 将权重参数初始化为均匀分布，范围在[-stdv, stdv]之间
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)  ## 如果使用偏置项，则将偏置参数也初始化为均匀分布，范围在[-stdv, stdv]之间

    def forward(self, input, adj):
        ## 这一步是为了将节点的特征进行线性变换，得到支持矩阵support，input的shape是（节点数，输入特征维度），weight的shape是（输入特征维度，输出特征维度），support的shape是（节点数，输出特征维度）
        support = torch.mm(input, self.weight)
        ## 得到支持矩阵之后，将邻接矩阵和支持矩阵进行乘法运算，得到输出矩阵。这一步是在图卷积中进行特征聚合的关键步骤
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias  ## 如果使用偏置项，则将输出矩阵加上偏置参数后返回
        else:
            return output  ## 如果不使用偏置项，则直接返回输出矩阵
        
    def __repr__(self):  ## 定义类的字符串表示方法，当打印类的实例时会调用该方法，返回一个描述类的信息的字符串
        return self.__class__.__name__ + ' (' + str(self.linear.in_features) + ' -> ' + str(self.linear.out_features) + ')'  ## 返回类的字符串表示，包含输入特征维度和输出特征维度的信息


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))  ## 定义权重矩阵W，大小为输入特征维度乘以输出特征维度
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  ## 使用Xavier均匀分布初始化权重矩阵W，gain参数设置为1.414
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  ## 定义注意力机制的参数a，大小为2倍输出特征维度乘以1
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  ## 使用Xavier均匀分布初始化注意力机制参数a，gain参数设置为1.414
        self.leakyrelu = nn.LeakyReLU(self.alpha)  ## 定义LeakyReLU激活函数，负斜率为alpha

        self.reset_parameters()  ## 调用reset_parameters方法初始化权重参数
    
    def reset_parameters(self):
        stdv = 1. / self.W.size(1)  ## 计算权重矩阵W的标准差，使用权重矩阵的列数进行计算
        self.W.data.uniform_(-stdv, stdv)  ## 将权重矩阵W初始化为均匀分布，范围在[-stdv, stdv]之间
        self.a.data.uniform_(-stdv, stdv)  ## 将注意力机制参数a初始化为均匀分布，范围在[-stdv, stdv]之间

    def forward(self, input, adj):
        """
        input: (N, in_features)  N是节点数，in_features是输入特征维度
        adj: (N, N)  邻接矩阵
        整个layer的forward其实就是首先将每个节点的特征进行线性变换，然后计算每个节点与其相邻节点之间的注意力权重，最后根据注意力权重对相邻节点的特征进行加权求和，得到每个节点的输出特征。
        """
        h = torch.mm(input, self.W)  ## 将输入特征矩阵input与权重矩阵W进行矩阵乘法，得到h，shape是（节点数，输出特征维度）
        N = h.size()[0]  ## 获取节点数N
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)  ## 构造注意力机制的输入a_input，shape是（节点数，节点数，2倍输出特征维度）
        ## (1,节点数，2倍输出特征维度)表示一个节点，与包括自己在内的所有节点的特征拼接在一起

        ## 得到每个节点与包括自己在内的其余节点的一个注意力值（未进行softmax）
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  ## 将a_input与注意力机制参数a进行矩阵乘法，并通过LeakyReLU激活函数处理，得到e，shape是（节点数，节点数）

        ## 
        zero_vec = -9e15 * torch.ones_like(e)  ## 创建一个与e形状相同的全为-9e15的张量，用于后续的掩码操作
        attention = torch.where(adj > 0, e, zero_vec)  ## 对e进行掩码操作，如果邻接矩阵adj中对应位置大于0，则保留e中的值，否则使用zero_vec中的值，得到attention，shape是（节点数，节点数）
        attention = F.softmax(attention, dim=1)  ## 对attention进行softmax操作，使得每行的元素和为1，得到最终的注意力权重矩阵attention，shape是（节点数，节点数）
        attention = F.dropout(attention, self.dropout, training=self.training)  ## 对注意力权重矩阵attention进行dropout操作，使用定义的dropout概率，并且只在训练模式下应用dropout
        h_prime = torch.matmul(attention, h)  ## 将注意力权重矩阵attention与h进行矩阵乘法，得到h_prime，shape是（节点数，输出特征维度）

        if self.concat:
            return F.elu(h_prime)  ## 如果concat为True，则对h_prime进行ELU激活函数处理后返回
        else:
            return h_prime  ## 如果concat为False，则直接返回h_prime


class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphSAGELayer, self).__init__()
        self.linear = nn.Linear(in_features * 2, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, self_features, neighbor_features):
        ## self_features 的形状是 (batch_size, in_features)，表示当前目标节点自身的特征。
        ## neighbor_features 的形状是 (batch_size, num_neighbors, in_features)，表示每个目标节点采样到的邻居特征。
        if neighbor_features.size(1) == 0:  ## 如果当前节点没有邻居，就退化为直接使用自身特征代替邻居聚合结果。
            neighbor_mean = self_features
        else:
            neighbor_mean = neighbor_features.mean(dim=1)  ## 对采样到的邻居特征在邻居维度上取平均，得到聚合后的邻居表示。
        combined = torch.cat([self_features, neighbor_mean], dim=1)  ## 将节点自身特征和邻居聚合特征拼接起来。
        return self.linear(combined)  ## 通过线性层把拼接结果映射到输出维度，得到当前层输出。

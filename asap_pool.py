import math

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_scatter import scatter_add, scatter_max

from torch_geometric.nn import GCNConv
from le_conv import LEConv
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax
from torch_geometric.nn.pool.topk_pool import topk

from torch_sparse import coalesce, transpose, spspmm


# torch.set_num_threads(1)


def StAS(index_A, value_A, index_S, value_S, device, N, kN):
    r"""StAS: a function which returns new edge weights for the pooled graph using the formula S^{T}AS"""
    # StAS: 一个函数，它使用公式S ^ {T}  AS为池图返回新的边权重
    index_A, value_A = coalesce(index_A, value_A, m=N, n=N)
    # COALESCE是一个函数， (expression_1, expression_2, ...,expression_n)依次参考各参数表达式，遇到非null值即停止并返回该值。
    # 如果所有的表达式都是空值，最终将返回一个空值。使用COALESCE在于大部分包含空值的表达式最终将返回空值。
    index_S, value_S = coalesce(index_S, value_S, m=N, n=kN)
    index_B, value_B = spspmm(index_A, value_A, index_S, value_S, N, N, kN)

    index_St, value_St = transpose(index_S, value_S, N, kN)
    index_B, value_B = coalesce(index_B, value_B, m=N, n=kN)
    # index_E, value_E = spspmm(index_St.cpu(), value_St.cpu(), index_B.cpu(), value_B.cpu(), kN, N, kN)
    index_E, value_E = spspmm(index_St, value_St, index_B, value_B, kN, N, kN)

    # return index_E.to(device), value_E.to(device)
    return index_E, value_E


def graph_connectivity(device, perm, edge_index, edge_weight, score, ratio, batch, N):
    r"""graph_connectivity: is a function which internally calls StAS func to maintain graph connectivity"""
    # graph_connectivity: 是一个函数，它在内部调用StAS函数以维护图形连接“”
    kN = perm.size(0)
    perm2 = perm.view(-1, 1)

    # mask contains bool mask of edges which originate from perm (selected) nodes #mask包含源自perm（选定）节点的边的布尔掩码
    mask = (edge_index[0] == perm2).sum(0, dtype=torch.bool)
    # mask:用选定的图像，图形或物体，对处理的图像（全部或局部）进行遮挡，来控制图像处理的区域或处理过程。用于覆盖的特定图像或物体称为掩模或模板。
    # 光学图像处理中，掩模可以足胶片，滤光片等。
    # create the S
    S0 = edge_index[1][mask].view(1, -1)
    S1 = edge_index[0][mask].view(1, -1)
    index_S = torch.cat([S0, S1], dim=0)
    value_S = score[mask].detach().squeeze()

    # relabel for pooling ie: make S [N x kN] 池的重新标记即：使S[N x kN]
    n_idx = torch.zeros(N, dtype=torch.long)
    n_idx[perm] = torch.arange(perm.size(0))
    index_S[1] = n_idx[index_S[1]]

    # create A
    index_A = edge_index.clone()
    if edge_weight is None:
        value_A = value_S.new_ones(edge_index[0].size(0))
    else:
        value_A = edge_weight.clone()

    fill_value = 1
    index_E, value_E = StAS(index_A, value_A, index_S, value_S, device, N, kN)
    index_E, value_E = remove_self_loops(edge_index=index_E, edge_attr=value_E)
    index_E, value_E = add_remaining_self_loops(index_E, value_E,
                                                fill_value=fill_value, num_nodes=kN)

    return index_E, value_E


class ASAP_Pooling(torch.nn.Module):

    def __init__(self, in_channels, ratio, dropout_att=0, negative_slope=0.2):
        super(ASAP_Pooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout_att = dropout_att
        self.lin_q = Linear(in_channels, in_channels)
        self.gat_att = Linear(2 * in_channels, 1)
        self.gnn_score = LEConv(self.in_channels,
                                1)  # gnn_score: uses LEConv to find cluster fitness scores gnn_score：使用LEConv查找集群适应度得分
        self.gnn_intra_cluster = GCNConv(self.in_channels,
                                         self.in_channels)  # gnn_intra_cluster: uses GCN to account for intra cluster properties, e.g., edge-weights
        # gnn_intra_cluster：使用GCN说明簇内属性，例如边缘权重
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.gat_att.reset_parameters()
        self.gnn_score.reset_parameters()
        self.gnn_intra_cluster.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # NxF
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # Add Self Loops 添加自我循环
        fill_value = 1
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight,
                                                           fill_value=fill_value, num_nodes=num_nodes.sum())

        N = x.size(0)  # total num of nodes in batch 批处理中的节点总数

        # ExF
        x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x_pool_j = x_pool[edge_index[1]]
        x_j = x[edge_index[1]]

        # ---Master query formation--- 主查询格式
        # NxF
        X_q, _ = scatter_max(x_pool_j, edge_index[0], dim=0)
        # NxF
        M_q = self.lin_q(X_q)
        # ExF
        M_q = M_q[edge_index[0].tolist()]

        score = self.gat_att(torch.cat((M_q, x_pool_j), dim=-1))
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[0], num_nodes=num_nodes.sum())

        # Sample attention coefficients stochastically.  随机抽样注意力系数。
        score = F.dropout(score, p=self.dropout_att, training=self.training)
        # ExF
        v_j = x_j * score.view(-1, 1)
        # ---Aggregation---聚合
        # NxF
        out = scatter_add(v_j, edge_index[0], dim=0)

        # ---Cluster Selection 群集选择
        # Nx1
        fitness = torch.sigmoid(self.gnn_score(x=out, edge_index=edge_index)).view(-1)
        perm = topk(x=fitness, ratio=self.ratio, batch=batch)
        x = out[perm] * fitness[perm].view(-1, 1)

        # ---Maintaining Graph Connectivity  维护图形连接
        batch = batch[perm]
        edge_index, edge_weight = graph_connectivity(
            device=x.device,
            perm=perm,
            edge_index=edge_index,
            edge_weight=edge_weight,
            score=score,
            ratio=self.ratio,
            batch=batch,
            N=N)

        return x, edge_index, edge_weight, batch, perm

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__, self.in_channels, self.ratio)

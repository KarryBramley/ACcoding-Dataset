from math import floor, sqrt
import random
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

def schlichtkrull_std(tensor, gain):
    """
    a = \text{gain} \times \frac{3}{\sqrt{\text{fan\_in} + \text{fan\_out}}}
    """
    fan_in, fan_out = tensor.shape[0], tensor.shape[1]
    return gain * 3.0 / sqrt(float(fan_in + fan_out))


def schlichtkrull_normal_(tensor, gain=1.):
    """Fill the input `Tensor` with values according to the Schlichtkrull method, using a normal distribution."""
    std = schlichtkrull_std(tensor, gain)
    with torch.no_grad():
        return tensor.normal_(0.0, std)


def schlichtkrull_uniform_(tensor, gain=1.):
    """Fill the input `Tensor` with values according to the Schlichtkrull method, using a uniform distribution."""
    std = schlichtkrull_std(tensor, gain)
    with torch.no_grad():
        return tensor.uniform_(-std, std)


def select_b_init(init):
    """Return functions for initialising biases"""
    init = init.lower()
    if init in ['zeros', 'zero', 0]:
        return torch.nn.init.zeros_
    elif init in ['ones', 'one', 1]:
        return torch.nn.init.ones_
    elif init == 'uniform':
        return torch.nn.init.uniform_
    elif init == 'normal':
        return torch.nn.init.normal_
    else:
        raise NotImplementedError(f'{init} initialisation has not been implemented!')


def select_w_init(init):
    """Return functions for initialising weights"""
    init = init.lower()
    if init in ['glorot-uniform', 'xavier-uniform']:
        return torch.nn.init.xavier_uniform_
    elif init in ['glorot-normal', 'xavier-normal']:
        return torch.nn.init.xavier_normal_
    elif init == 'schlichtkrull-uniform':
        return schlichtkrull_uniform_
    elif init == 'schlichtkrull-normal':
        return schlichtkrull_normal_
    elif init in ['normal', 'standard-normal']:
        return torch.nn.init.normal_
    elif init == 'uniform':
        return torch.nn.init.uniform_
    elif init == 'orthogonal':
        return torch.nn.init.orthogonal_
    else:
        raise NotImplementedError(f'{init} initialisation has not been implemented!')


def select_activation(function):
    """Return activation functions for outputs"""
    function = function.lower()
    if function == 'relu':
        return F.relu
    elif function == 'elu':
        return F.elu
    elif function == 'lrelu':
        return F.leaky_relu
    elif function == 'selu':
        return F.selu
    else:
        raise NotImplementedError(f'{function} activation function has not been implemented!')


def drop_edges(triples, num_nodes, general_edo, self_loop_edo):
    """ Performs edge dropout by actually removing the triples """
    general_keep = 1.0 - general_edo
    self_loop_keep = 1.0 - self_loop_edo

    # Notes: self-loop triples were appended to the end of the list in add_inverse_and_self
    nt = triples.size(0) - num_nodes

    general_keep_ind = random.sample(range(nt), k=int(floor(general_keep * nt)))
    self_loop_keep_ind = random.sample(range(nt, nt + num_nodes), k=int(floor(self_loop_keep * num_nodes)))
    ind = general_keep_ind + self_loop_keep_ind

    return triples[ind, :]


def add_inverse_and_self(triples, nodes_list, num_rels, self_loop_keep_prob, device='cpu'):
    """ Adds inverse relations and self loops to a tensor of triples """

    num_nodes = len(nodes_list)

    # Swap around head and tail. Create new relation ids for inverse relations.
    inverse_relations = torch.cat(
        [triples[:, [-1]], triples[:, [1]] + num_rels, triples[:, [2]], triples[:, [3]], triples[:, [0]]], dim=1)
    assert inverse_relations.size() == triples.size()

    # Create a new relation id for self loop relation. 给self-loop创建一个新的关系id——2*num+rels，然后构建三元组
    all = torch.tensor(nodes_list, device=device)[:, None]
    rid = torch.empty(size=(num_nodes, 1), device=device, dtype=torch.long).fill_(2 * num_rels)
    weight = torch.zeros(num_nodes, device=device, dtype=torch.long)[:, None]  # self-loop的权重都设为0，方便后面计算c_ij时不产生干扰
    time = torch.zeros(num_nodes, device=device, dtype=torch.long)[:, None]  # self-loop的时间设为0，让它排在序列的第一位
    self_loops = torch.cat([all, rid, weight, time, all], dim=1)
    assert self_loops.size() == (num_nodes, 5)

    # Apply edge dropout
    # torch.bernoulli，从伯努利分布中抽取二进制随机数(0或1),就是把原来的0~1之间矩阵值随机用0和1替代，如果原本值是1则新的值一定是1，0则一定是0.
    mask = torch.bernoulli(torch.empty(size=(num_nodes,), dtype=torch.float, device=device).fill_(
        self_loop_keep_prob)).to(torch.bool)
    self_loops = self_loops[mask, :]

    # Note: Self-loops are appended to the end and this makes it easier to apply different edge dropout rates.
    return torch.cat([triples, inverse_relations, self_loops], dim=0)


def split_spo(triples):
    """ Splits tensor into subject, predicate and object """
    if len(triples.shape) == 2:
        return triples[:, 0], triples[:, 1], triples[:, -1]
    else:
        return triples[:, :, 0], triples[:, :, 1], triples[:, :, -1]


def generate_neighbors_lists(triples, nodes_list, num_nodes):
    """ 为每个节点找到它的邻居节点列表（时序序列，带时间和结果）.
        output： 两层list，第一层list表示每个节点，第二层长度等于该节点的邻居节点序列长度"""

    neighbor_list = [[] for _ in range(num_nodes)]
    for s, p, w, t, o in triples:
        s, p, w, t, o = (s.item(), p.item(), w.item(), t.item(), o.item())
        neighbor_list[s].append([o, p, w, t])

    sorted_neighbor_list = [sorted(neighbor_list[nid], key=lambda x:x[3]) for nid in nodes_list]

    return sorted_neighbor_list


def select_max_time_neighbors(features, s_list, neighbors_len, num_nodes):
    """
        选择每个邻居中时间最晚的压缩方法，不保存成[num_nodes, num_nodes, emb_dim]的稀疏矩阵
        而是两个分别表示头和尾节点的indices和一个有内容的embedding矩阵，位置一一对应。
        长度都是unique_edge_num，合并了相同两个节点的重复边。
        输入：
        features: [batch_size, max_neighbor_num, emb_dim+2]，最里层为节点邻居[o, p，embedding]
        s_list: [batch_size, ]，这个batch中的s节点id
        neighbors_len：[batch_size, ]，每个s节点对应的邻居序列长度
        num_nodes: 节点总数，用于计算p和o联合的纵坐标indices_po
    """

    # 因为要找最晚时间，所以先用dict过渡一下，相同尾节点的位置直接覆盖。
    hidden_dict = {}

    for i, h_matrix in enumerate(features):
        s = s_list[i]
        hidden_dict[s] = {}
        for j in range(neighbors_len[i]):
            h = h_matrix[j]
            o = h[0].item()
            r = h[1].item()
            w = h[2].item()
            hidden_dict[s][r * num_nodes + o] = (r, w, h[3:])

    indices_s = torch.tensor([s for s, neighbors in hidden_dict.items() for _ in range(len(neighbors))]).long()
    indices_po = torch.tensor([po for s, neighbors in hidden_dict.items() for po, _ in neighbors.items()]).long()
    relations = torch.tensor([r for s, neighbors in hidden_dict.items() for _, (r, _, _) in neighbors.items()]).long()
    weights = torch.tensor([w for s, neighbors in hidden_dict.items() for _, (_, w, _) in neighbors.items()]).long()
    embeddings = torch.tensor([emb.tolist() for s, neighbors in hidden_dict.items() for _, (_, _, emb) in neighbors.items()])

    assert indices_s.size() == indices_po.size() == relations.size() == weights.size()

    return indices_s, indices_po, relations, weights, embeddings


def inverse_function(x):
    if x == 0:
        return 1.0 * x
    else:
        return 1.0 / x


def generate_adj_coefficient(triples, num_nodes, num_relations):
    """
        计算邻接矩阵adj，并用1/c_ij来填充，作为聚合函数中每一项的系数。
    """
    indices_s, weights = triples[:, 0], triples[:, 2]
    indices_po = triples[:, 1] * num_nodes + triples[:, -1]

    adj_shape = (num_nodes, num_nodes * num_relations)
    adj = sp.coo_matrix((weights, (indices_s, indices_po)), shape=adj_shape)

    # 计算每个节点对应的邻居数（除自己本身外，通过将self-loop的weight设为0来排除）
    row_sum = np.array(adj.sum(axis=1)).flatten()

    # 计算倒数，1/c_ij
    inv_func = np.vectorize(inverse_function)
    r_inv = inv_func(row_sum)
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)

    # 将每个节点自身的系数设为1，以实现聚合函数的后半部分
    for i in range(num_nodes):
        # self-loop关系根据定义是最后一个relation，所以relation_id = num_relations - 1（这里的num_relations是plus半的）
        adj[i, num_nodes * (num_relations - 1) + i] = 1

    # 转换为dok_matrix的稀疏矩阵储存方式，便于直接定位到每个非零元素
    adj = adj.todok()
    return adj

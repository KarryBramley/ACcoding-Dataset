from math import floor, sqrt
import random
import torch


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
    else:
        raise NotImplementedError(f'{init} initialisation has not been implemented!')

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

def sum_sparse(indices, values, size, row_normalisation=True, device='cpu'):
    """
    Sum the rows or columns of a sparse matrix, and redistribute the
    results back to the non-sparse row/column entries
    Arguments are interpreted as defining sparse matrix.
    把描述稀疏矩阵的参数拿来构建一个稀疏矩阵，然后计算行或列的和并返回这个新的、描述行或列加和的非稀疏矩阵

    Source: https://github.com/pbloem/gated-rgcn/blob/1bde7f28af8028f468349b2d760c17d5c908b58b/kgmodels/util/util.py#L304
    """

    assert len(indices.size()) == len(values.size()) + 1

    k, r = indices.size()

    if not row_normalisation:
        # Transpose the matrix for column-wise normalisation，dim=1维度里，二元组位置前后交换了一下
        indices = torch.cat([indices[:, 1:2], indices[:, 0:1]], dim=1)
        size = size[1], size[0]

    ones = torch.ones((size[1], 1), device=device)  # [size[1], 1]

    # 这里正经构建了稀疏矩阵，torch.sparse就是用values的值(也就是1)来填充indices对应的位置，转成二元组的indices也是因为稀疏矩阵是二维的
    # value: [size[0], size[1]]
    if device == 'cuda':
        values = torch.cuda.sparse.FloatTensor(indices.t(), values, torch.Size(size))
    else:
        values = torch.sparse.FloatTensor(indices.t(), values, torch.Size(size))
    sums = torch.spmm(values, ones)   # 稀疏矩阵乘法，乘ones相当于每行加和，结果[size[0], 1]
    sums = sums[indices[:, 0], 0]     # 查找indices里每个（实体+关系）对应的sum结果，也就是和指定实体以这种关系连接的实体有几个。[indices.size[0],]
    return sums.view(k)


def generate_inverses(triples, num_rels):
    """ Generates nverse relations """

    # Swap around head and tail. Create new relation ids for inverse relations.
    inverse_relations = torch.cat([triples[:, -1, None], triples[:, 1, None] + num_rels, triples[:, 2, None], triples[:, 0, None]], dim=1)
    assert inverse_relations.size() == triples.size()

    return inverse_relations


def generate_self_loops(triples, num_nodes, num_rels, self_loop_keep_prob, device='cpu'):
    """ Generates self-loop triples and then applies edge dropout """

    # Create a new relation id for self loop relation. 给self-loop创建一个新的关系id——2*num+rels，然后构建三元组
    all = torch.arange(num_nodes, device=device)[:, None]
    id  = torch.empty(size=(num_nodes, 1), device=device, dtype=torch.long).fill_(2*num_rels)
    weight = torch.ones(num_nodes, device=device, dtype=torch.long)[:, None]  # self-loop的权重都设为1
    self_loops = torch.cat([all, id, weight, all], dim=1)
    assert self_loops.size() == (num_nodes, 4)

    # Apply edge dropout
    # torch.bernoulli，从伯努利分布中抽取二进制随机数(0或1),就是把原来的0~1之间矩阵值随机用0和1替代，如果原本值是1则新的值一定是1，0则一定是0.
    mask = torch.bernoulli(torch.empty(size=(num_nodes,), dtype=torch.float, device=device).fill_(
        self_loop_keep_prob)).to(torch.bool)
    self_loops = self_loops[mask, :]

    return torch.cat([triples, self_loops], dim=0)


def add_inverse_and_self(triples, num_nodes, num_rels, device='cpu'):
    """ Adds inverse relations and self loops to a tensor of triples """

    # Swap around head and tail. Create new relation ids for inverse relations.
    inverse_relations = torch.cat([triples[:, 2, None], triples[:, 1, None] + num_rels, triples[:, 0, None]], dim=1)
    assert inverse_relations.size() == triples.size()

    # Create a new relation id for self loop relation.
    all = torch.arange(num_nodes, device=device)[:, None]
    id  = torch.empty(size=(num_nodes, 1), device=device, dtype=torch.long).fill_(2*num_rels)
    self_loops = torch.cat([all, id, all], dim=1)
    assert self_loops.size() == (num_nodes, 3)

    # Note: Self-loops are appended to the end and this makes it easier to apply different edge dropout rates.
    return torch.cat([triples, inverse_relations, self_loops], dim=0)

def stack_matrices(triples, num_nodes, num_rels, vertical_stacking=True, device='cpu'):
    """
    Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all
    relations are stacked vertically).
    也就是将三元组压缩成二元组，[triple_size, 3] -> [triple_size, 2]
    """
    assert triples.dtype == torch.long

    r, n = num_rels, num_nodes
    size = (r * n, n) if vertical_stacking else (n, r * n)

    fr, to = triples[:, 0], triples[:, -1]   # 源节点和目标节点，都是[triple_size,]
    offset = triples[:, 1] * n  # 关系乘节点数，[triple_size,]

    # 如果vertical_stacking，则把头和关系(offset)加到一起，尾保持不变；否则就把尾和关系加起来，头不变
    if vertical_stacking:
        fr = offset + fr
    else:
        to = offset + to

    # 把压缩后头和尾再连接起来
    indices = torch.cat([fr[:, None], to[:, None]], dim=1).to(device)   # [triple_size, 2]

    assert indices.size(0) == triples.size(0)
    assert indices[:, 0].max() < size[0], f'{indices[0, :].max()}, {size}, {r}'
    assert indices[:, 1].max() < size[1], f'{indices[1, :].max()}, {size}, {r}'

    return indices, size

def block_diag(m):
    """
    Source: https://gist.github.com/yulkang/2e4fc3061b45403f455d7f4c316ab168
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    """

    device = 'cuda' if m.is_cuda else 'cpu'  # Note: Using cuda status of m as proxy to decide device

    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    dim = m.dim()
    n = m.shape[-3]

    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]

    m2 = m.unsqueeze(-2)

    eye = attach_dim(torch.eye(n, device=device).unsqueeze(-2), dim - 3, 1)

    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(torch.Size([1] * n_dim_to_prepend) + v.shape + torch.Size([1] * n_dim_to_append))

def split_spo(triples):
    """ Splits tensor into subject, predicate and object """
    if len(triples.shape) == 2:
        return triples[:, 0], triples[:, 1], triples[:, -1]
    else:
        return triples[:, :, 0], triples[:, :, 1], triples[:, :, -1]

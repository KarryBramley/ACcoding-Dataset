import numpy as np
from random import sample
import torch

def select_sampling(method):
    method = method.lower()
    if method == 'uniform':
        return uniform_sampling
    elif method == 'edge-neighborhood':
        return edge_neighborhood
    else:
        raise NotImplementedError(f'{method} sampling method has not been implemented!')


def uniform_sampling(train_triples, num_nodes, sample_size=30000):
    """Random uniform sampling"""
    return sample(train_triples, sample_size)


def edge_neighborhood(train_triples, num_nodes, sample_size=30000):
    """Edge neighborhood sampling
    从有邻居节点的节点中去采样"""

    # TODO: Clean this up
    # entities = {v: k for k, v in entities.items()}
    adj_list = [[] for _ in range(num_nodes)]
    # 直接用triple的源节点和目标节点构建的邻接list，节点id为index，与它有连接的所有节点是再组成一个list，并且有triple的位置
    for i, triplet in enumerate(train_triples):
        adj_list[triplet[0]].append([i, triplet[-1]])
        adj_list[triplet[-1]].append([i, triplet[0]])
    # 用上面那个list可以很方便的算出每个节点的degree
    degrees = np.array([len(a) for a in adj_list])  # [entities, ]
    adj_list = [np.array(a) for a in adj_list]

    edges = np.zeros((sample_size), dtype=np.int32)  # [sample_size,], value: 0

    sample_counts = np.array([d for d in degrees])  # [entities,], value: degree
    picked = np.array([False for _ in train_triples])  # [train_triples,], value: False
    seen = np.array([False for _ in degrees])  # [entities,], value: False

    for i in range(0, sample_size):
        # 乘完都是0...
        weights = sample_counts * seen

        # degree是0的位置设为0，其他设为1
        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights) if np.sum(weights) != 0 else (weights)
        # print(i, probabilities, len(probabilities), sum(probabilities))
        # np.random.choice，从给定一维数组中生成随机样本，p是一维数组中每一项的概率，如果没有给定则默认是均匀分布的
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]), p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    edges = [train_triples[e] for e in edges]
    return edges


def extract_nodes(triples):
    s, o = set(), set()
    for t in triples:
        s.add(t[0])
        o.add(t[-1])
    nodes = s | o
    return list(nodes)


def corrupt(batch, num_nodes, head_corrupt_prob, device='cpu'):
    """Corrupts the negatives of a batch of triples. Randomly corrupts either heads or tails."""
    bs, ns, _ = batch.size()  # batch_size, neg_sample_rate

    # new entities to insert
    corruptions = torch.randint(size=(bs * ns,), low=0, high=num_nodes, dtype=torch.long, device=device)

    # boolean mask for entries to corrupt
    mask = torch.bernoulli(torch.empty(
        size=(bs, ns, 1), dtype=torch.float, device=device).fill_(head_corrupt_prob)).to(torch.bool)
    zeros = torch.zeros(size=(bs, ns, 1), dtype=torch.bool, device=device)
    # 得到用来替换头和尾的mask矩阵(用~取反以保证头尾总有一个被替换的)，mask为True的位置替换成corruption生成的新值，p和w不用管，所以都是0（False）
    mask = torch.cat([mask, zeros, zeros, zeros, ~mask], dim=2)
    batch[mask] = corruptions

    return batch.view(bs * ns, -1)


def negative_sampling(positive_triples, entity_dictionary, neg_sample_rate):
    """ Generates a set of negative samples by corrupting triples """

    all_triples = np.array(positive_triples)
    s = np.resize(all_triples[:, 0], (len(positive_triples) * neg_sample_rate,))
    p = np.resize(all_triples[:, 1], (len(positive_triples) * neg_sample_rate,))
    o = np.random.randint(low=0, high=len(entity_dictionary), size=(len(positive_triples) * neg_sample_rate,))
    negative_triples = np.stack([s, p, o], axis=1)

    return negative_triples.tolist()


def corrupt_heads(entity_dictionary, p, o):
    """ Generate a list of candidate triples by replacing the head with every entity for each test triplet """
    return [(s, p, o) for s in range(len(entity_dictionary))]


def corrupt_tails(s, p, entity_dictionary):
    """ Generate a list of candidate triples by replacing the tail with every entity for each test triplet """
    return [(s, p, o) for o in range(len(entity_dictionary))]


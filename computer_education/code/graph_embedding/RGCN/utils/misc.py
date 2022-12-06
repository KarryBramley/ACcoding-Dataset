from functools import cmp_to_key
from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np
from random import sample
import torch
import tqdm
import os
import math
import torch.nn.functional as F


def create_experiment(name='exp', database=None):
    """ Create Scared experiment object for experiment logging """
    ex = Experiment(name)

    atlas_user = os.environ.get('MONGO_DB_USER')
    atlas_password = os.environ.get('MONGO_DB_PASS')
    atlas_host = os.environ.get('MONGO_DB_HOST')

    # Add remote MongoDB observer, only if environment variables are set
    if atlas_user and atlas_password and atlas_host:
        ex.observers.append(MongoObserver(
            url=f"mongodb+srv://{atlas_user}:{atlas_password}@{atlas_host}",
            db_name=database))
    return ex


#######################################################################################################################
# Relation Prediction Utils
#######################################################################################################################

def generate_true_dict(all_triples):
    """ Generates a pair of dictionaries containing all true tail and head completions """
    heads, tails = {(p, o): [] for _, p, _, o in all_triples}, {(s, p): [] for s, p, _, _ in all_triples}

    for s, p, _, o in all_triples:
        heads[p, o].append(s)
        tails[s, p].append(o)

    return heads, tails


def filter_scores_(scores, batch, true_triples, head=True):
    """ Filters a score matrix by setting the scores of known non-target true triples to -inf """

    # TODO Clean this up
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    indices = []  # indices of triples whose scores should be set to -infty

    heads, tails = true_triples

    for i, (s, p, _, o) in enumerate(batch):
        s, p, o = triple = (s.item(), p.item(), o.item())
        if head:
            indices.extend([(i, si) for si in heads[p, o] if si != s])
        else:
            indices.extend([(i, oi) for oi in tails[s, p] if oi != o])
        # -- We add the indices of all know triples except the one corresponding to the target triples.

    indices = torch.tensor(indices, device=device)

    scores[indices[:, 0], indices[:, 1]] = float('-inf')

def evaluate_old(model, graph, test_set, true_triples, num_nodes, batch_size=16, hits_at_k=[1, 3, 10],
             filter_candidates=True, verbose=True):
    """ Evaluates a triple scoring model. Does the sorting in a single, GPU-accelerated operation. """

    # TODO Clean this up
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rng = tqdm.trange if verbose else range
    ranks = []
    for head in [True, False]:  # head or tail prediction

        for fr in rng(0, len(test_set), batch_size):
            to = min(fr + batch_size, len(test_set))

            batch = test_set[fr:to, :].to(device=device)
            bn, _ = batch.size()

            # compute the full score matrix (filter later)
            # head=true：base-[r,o], target-[s]；head=false：base-[s,r], target-[r]
            bases = torch.cat((batch[:, 1][:, None], batch[:, -1][:, None]), dim=1) if head else batch[:, :2]
            targets = batch[:, 0] if head else batch[:, -1]

            # collect the triples for which to compute scores
            # .expand用来把本来那个维度的向量复制指定次
            # 在dim=1的维度，将[r,o]或[s,r]复制num_nodes次
            bexp = bases.view(bn, 1, 2).expand(bn, num_nodes, 2)  # [batch_size, num_nodes, 2]
            # 在dim=0，将[[0],[1],[2]...]复制bn次
            ar = torch.arange(num_nodes, device=device).view(1, num_nodes, 1).expand(bn, num_nodes,
                                                                                     1)  # [batch_size, num_nodes, 1]
            # 把index和bases concat在一起，是为了计算从0~num_nodes每个实体的得分
            toscore = torch.cat([ar, bexp] if head else [bexp, ar], dim=2)  # [batch_size, num_nodes, 3]
            assert toscore.size() == (bn, num_nodes, 3)

            # score: [batch_num, node_num]，test集每个triple的预测结果，这个triple的s或o与其他所有实体的乘积
            scores, _ = model(graph, toscore)
            assert scores.size() == (bn, num_nodes)

            # filter out the true triples that aren't the target
            if filter_candidates:
                filter_scores_(scores, batch, true_triples, head=head)

            # Select the true scores, and count the number of values larger than than
            # 从得到的scores里面找出真实的实体对应的得分
            true_scores = scores[torch.arange(bn, device=device), targets]
            # 计算比true实体的得分高的实体个数
            raw_ranks = torch.sum(scores > true_scores.view(bn, 1), dim=1, dtype=torch.long)
            # -- This is the "optimistic" rank (assuming it's sorted to the front of the ties)
            # 再额外统计一下得分和true entity一样的实体数，通常都是1（就是true entity本身）
            num_ties = torch.sum(scores == true_scores.view(bn, 1), dim=1, dtype=torch.long)

            # Account for ties (put the true example halfway down the ties)
            # num_ties-1排除掉true entity自己，若是有一样的，就给它对个半
            branks = raw_ranks + (num_ties - 1) // 2

            # 把这个batch的排名（都+1）续到ranks列表上去
            ranks.extend((branks + 1).tolist())

    # mrr和rank的值成反比，若rank值越小，说明比true entity得分高的实体越少，那这个预测的结果就越好，mrr越大
    mrr = sum([1.0 / rank for rank in ranks]) / len(ranks)

    # hits是用来计算rank值比k小的个数，hits值越大，结果越好
    hits = []
    for k in hits_at_k:
        hits.append(sum([1.0 if rank <= k else 0.0 for rank in ranks]) / len(ranks))
    return mrr, tuple(hits), ranks


def evaluate(test_set, embeddings, batch_size=16, verbose=True, log=None, save_dir=None):
    """ Evaluates a triple scoring model. Does the sorting in a single, GPU-accelerated operation. """

    # TODO Clean this up
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_embs = {x.item(): embeddings[x] for x in test_set[:, 0].unique()}
    Y_embs = {y.item(): embeddings[y] for y in test_set[:, -1].unique()}

    rng = tqdm.trange if verbose else range
    print('evaluating...')

    f1, map, mrr, mndcg = top_n(test_set, X_embs, Y_embs, 10)
    rec_metrics = {'F1': round(f1, 4),
                   'MAP': round(map, 4),
                   'MRR': round(mrr, 4),
                   'MNDCG': round(mndcg, 4)}

    # rec_metrics = {'F1': [],
    #                'MAP': [],
    #                'MRR': [],
    #                'MNDCG': []}
    #
    # for fr in rng(0, len(test_set), batch_size):
    #     # print('for fr in batch:', time.time())
    #     to = min(fr + batch_size, len(test_set))
    #
    #     batch = test_set[fr:to, :]
    #     bn, _ = batch.size()
    #
    #     f1, map, mrr, mndcg = top_n(batch, X_embs, Y_embs, 10)
    #     rec_metrics['F1'].append(f1)
    #     rec_metrics['MAP'].append(map)
    #     rec_metrics['MRR'].append(mrr)
    #     rec_metrics['MNDCG'].append(mndcg)
    #
    # # 求一下平均
    # rec_metrics = {k: round(sum(np.array(v)) / len(v), 4) for k, v in rec_metrics.items()}
    return rec_metrics


def top_n(test, X_embs, Y_embs, top_n):
    test_x = test[:, 0]  # [triple_num,]
    test_y = test[:, -1]  # [triple_num,]
    weight = test[:, 2]  # [triple_num, ]

    rates_dict = {}
    scores_dict = {}

    rng = tqdm.trange

    for i in rng(len(test_x)):
        x = test_x[i].item()
        y = test_y[i].item()
        X = X_embs[x]

        if rates_dict.get(x) is None:
            rates_dict[x] = {}
        if scores_dict.get(x) is None:
            scores_dict[x] = {}
            for j, Y in Y_embs.items():
                # Y = np.array(embeddings[j.item()].cpu())
                scores_dict[x][j] = X.dot(Y)

        rates_dict[x][y] = rates_dict[x][y] + weight[i].item() if rates_dict[x].get(y) else weight[i].item()

    precision_list = []
    recall_list = []
    ap_list = []
    ndcg_list = []
    rr_list = []

    # for x, emb in X_embs.items():
    for x, emb in X_embs.items():
        # x = x.item()
        tmp_r = sorted(scores_dict[x].items(), key=cmp_to_key(lambda m, n: cmp(m[1], n[1])),
                       reverse=True)[: min(top_n, len(scores_dict[x]))]
        tmp_t = sorted(rates_dict[x].items(), key=cmp_to_key(lambda m, n: cmp(m[1], n[1])),
                       reverse=True)[: min(top_n, len(rates_dict[x]))]
        tmp_r_list = []
        tmp_t_list = []
        for (item, rate) in tmp_r:
            tmp_r_list.append(item)
        for (item, rate) in tmp_t:
            tmp_t_list.append(item)

        pre, rec = precision_and_recall(tmp_r_list, tmp_t_list)
        ap = AP(tmp_r_list, tmp_t_list)
        rr = RR(tmp_r_list, tmp_t_list)
        ndcg = nDCG(tmp_r_list, tmp_t_list)
        precision_list.append(pre)
        recall_list.append(rec)
        ap_list.append(ap)
        rr_list.append(rr)
        ndcg_list.append(ndcg)

    precision = sum(precision_list) / len(precision_list)
    recall = sum(recall_list) / len(recall_list)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    map = sum(ap_list) / len(ap_list)
    mrr = sum(rr_list) / len(rr_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return f1, map, mrr, mndcg


def select_sampling(method):
    method = method.lower()
    if method == 'uniform':
        return uniform_sampling
    elif method == 'edge-neighborhood':
        return edge_neighborhood
    else:
        raise NotImplementedError(f'{method} sampling method has not been implemented!')


def uniform_sampling(graph, sample_size=300, entities=None, train_triplets=None):
    """Random uniform sampling"""
    return sample(graph, sample_size)


def edge_neighborhood(train_triples, sample_size=300, entities=None):
    """Edge neighborhood sampling
    从有邻居节点的节点中去采样"""

    # TODO: Clean this up
    entities = {v: k for k, v in entities.items()}
    adj_list = [[] for _ in entities]
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
    mask = torch.cat([mask, zeros, zeros, ~mask], dim=2)
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


def filter_triples(candidate_triples, all_triples, correct_triple):
    """ Filter out candidate_triples that are present in all_triples, but keep correct_triple """
    return [triple for triple in set(candidate_triples) if not triple in all_triples or triple == correct_triple]


def compute_mrr(rank):
    """ Compute Mean Reciprocal Rank for a given list of ranked triples """
    return 1.0 / rank


def compute_hits(rank, k):
    """ Compute Precision at K for a given list of ranked triples """
    if k == 1:
        return 1 if rank == k else 0
    else:
        return 1 if rank <= k else 0


def rank_triple(scores, candidate_triples, correct_triple):
    """ Finds rank of the correct triple after sorting candidates by their scores """
    sorted_candidates = [tuple(i[0]) for i in
                         sorted(zip(candidate_triples.tolist(), scores.tolist()), key=lambda i: -i[1])]
    rank = sorted_candidates.index(correct_triple) + 1
    return rank


def compute_metrics(scores, candidates, correct_triple, k=None):
    """ Returns MRR and Hits@k (k=1,3,10) values for a given triple prediction """
    if k is None:
        k = [1, 3, 10]

    rank = rank_triple(scores, candidates, correct_triple)
    mrr = compute_mrr(rank)
    hits_at_k = {i: compute_hits(rank, i) for i in k}

    return mrr, hits_at_k


def cmp(x, y):
    if x > y:
        return 1
    elif x < y:
        return -1
    else:
        return 0


def nDCG(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    return dcg / idcg


def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg


def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0


def RR(ranked_list, ground_list):
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0


def precision_and_recall(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits / (1.0 * len(ranked_list))
    rec = hits / (1.0 * len(ground_list))
    return pre, rec

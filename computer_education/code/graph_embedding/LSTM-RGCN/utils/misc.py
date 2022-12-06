from functools import cmp_to_key
from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import os
import math
import time
from torch.nn import functional as F


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
    heads, tails = {(p, o): [] for _, p, _, _, o in all_triples}, {(s, p): [] for s, p, _, _, _ in all_triples}

    for s, p, _, _, o in all_triples:
        heads[p, o].append(s)
        tails[s, p].append(o)

    return heads, tails


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
        # print('x:', x)
        # print('x_scores:', scores_dict[x])
        # print('x_rates:', rates_dict[x])
        tmp_r = sorted(scores_dict[x].items(), key=cmp_to_key(lambda m, n: cmp(m[1], n[1])),
                       reverse=False)[: min(top_n, len(scores_dict[x]))]
        tmp_t = sorted(rates_dict[x].items(), key=cmp_to_key(lambda m, n: cmp(m[1], n[1])),
                       reverse=True)[: min(top_n, len(rates_dict[x]))]
        # print('tmp_r:', tmp_r)
        # print('tmp_t:', tmp_t)

        tmp_r_list = [item for item, _ in tmp_r]
        tmp_t_list = [item for item, _ in tmp_t]

        pre, rec = precision_and_recall(tmp_r_list, tmp_t_list)
        # print('pre:', pre, 'rec:', rec, '\n')
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


def filter_triples(candidate_triples, all_triples, correct_triple):
    """ Filter out candidate_triples that are present in all_triples, but keep correct_triple """
    return [triple for triple in set(candidate_triples) if not triple in all_triples or triple == correct_triple]


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

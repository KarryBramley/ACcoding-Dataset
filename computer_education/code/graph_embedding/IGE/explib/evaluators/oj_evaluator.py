import os
import numpy as np
import pandas as pd
from functools import cmp_to_key
import math
import torch.nn.functional as functional
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, normalized_mutual_info_score


class OjEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate(self, emb_x, emb_y, eval_data='test', linear_model=None):

        # device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
        self.emb_x = torch.tensor(emb_x)
        self.emb_y = torch.tensor(emb_y)

        # test recommendation task
        test_x, test_y, test_rate = self.read_rec_data(eval_data)
        f1, mean_ap, mrr, mndcg = self.top_n(test_x, test_y, test_rate, self.config.top_n)
        # acc_score = self.classification(emb_y)

        rec_metrics = {'F1': round(f1, 4),
                       'MAP': round(mean_ap, 4),
                       'MRR': round(mrr, 4),
                       'MNDCG': round(mndcg, 4)}

        # test classification task
        # if eval_data != 'valid' and linear_model is not None:
        #     assert linear_model is not None
        #     self.cls_path = f'./data/{self.config.exp_name}/pid_label_{eval_data}.csv'
        #     self.label = pd.read_csv(self.cls_path)
        #     self.linear_model = linear_model
        #     micro_f1, macro_f1 = self.classification(emb_y)
        #     cls_metrics = {'MICRO_F1': round(micro_f1, 4),
        #                    'MACRO_F1': round(macro_f1, 4)}
        #
        #     return rec_metrics, cls_metrics

        return rec_metrics

    def read_rec_data(self, eval_data):
        root = f'./data/{self.config.exp_name}/'
        fname = f'{eval_data}_recommendation.csv'

        prefix = ''
        if self.config.use_mini:
            prefix = 'mini_'

        # self.test = pd.read_csv(os.path.join(root, fname))
        users, problems, rates = set(), set(), {}
        with open(os.path.join(root, prefix + fname), 'r', encoding='UTF-8') as fin:
            line = fin.readline()
            while line:
                user, problem, rate, _ = [int(s) for s in line.strip().split(',')]
                if rates.get(user) is None:
                    rates[user] = {}
                rates[user][problem] = float(rate)
                users.add(user)
                problems.add(problem)
                line = fin.readline()
        return users, problems, rates

    def top_n(self, test_x, test_y, test_rate, top_n):
        emb_x = self.emb_x
        emb_y = self.emb_y
        recommend_dict = {}
        # 模型得到的emb_x和emb_y矩阵都是按用户和题目id顺序排列的，所以直接emb_x[x]就表示id为x的用户的embedding
        for x in test_x:
            recommend_dict[x] = {}
            for y in test_y:
                X = emb_x[x]
                Y = emb_y[y]
                pre = X.dot(Y.T)
                recommend_dict[x][y] = float(pre)

        precision_list = []
        recall_list = []
        ap_list = []
        ndcg_list = []
        rr_list = []

        for x in test_x:
            # tmp_r就是最终推荐的内容，tmp_t用来评估
            tmp_r = sorted(recommend_dict[x].items(), key=cmp_to_key(lambda x, y: cmp(x[1], y[1])),
                           reverse=True)[0: min(len(recommend_dict[x]), top_n)]
            tmp_t = sorted(test_rate[x].items(), key=cmp_to_key(lambda x, y: cmp(x[1], y[1])),
                           reverse=True)[0:min(len(test_rate[x]), top_n)]

            tmp_r_list = [item for (item, rate) in tmp_r]
            tmp_t_list = [item for (item, rate) in tmp_t]
            # print('tmp_r:', tmp_r)
            # print('tmp_t:', tmp_t)
            # print('tmp_r_list:', tmp_r_list)
            # print('tmp_t_list:', tmp_t_list)
            # exit(0)
            # for (item, rate) in tmp_r:
            #     tmp_r_list.append(item)
            #
            # for (item, rate) in tmp_t:
            #     tmp_t_list.append(item)

            pre, rec = precision_and_recall(tmp_r_list, tmp_t_list)
            ap = AP(tmp_r_list, tmp_t_list)
            rr = RR(tmp_r_list, tmp_t_list)
            ndcg = nDCG(tmp_r_list, tmp_t_list)
            precision_list.append(pre)
            recall_list.append(rec)
            ap_list.append(ap)
            rr_list.append(rr)
            ndcg_list.append(ndcg)

        precison = sum(precision_list) / len(precision_list)
        recall = sum(recall_list) / len(recall_list)
        f1 = 2 * precison * recall / (precison + recall)
        map = sum(ap_list) / len(ap_list)
        mrr = sum(rr_list) / len(rr_list)
        mndcg = sum(ndcg_list) / len(ndcg_list)
        return f1, map, mrr, mndcg

    def classification(self, embd):

        x = embd[self.label.pid]
        y = self.label.label
        cls_weight = self.linear_model.weight.data
        cls_bias = self.linear_model.bias.data

        y_norm = functional.linear(torch.Tensor(x), cls_weight, cls_bias)
        y_pred = torch.argmax(y_norm, dim=1)

        micro_f1 = f1_score(y, y_pred, average='micro')
        macro_f1 = f1_score(y, y_pred, average='macro')

        return micro_f1, macro_f1


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
        # id就是u
        if id in ground_list:
            hits += 1
    pre = hits / (1.0 * len(ranked_list))
    rec = hits / (1.0 * len(ground_list))
    return pre, rec

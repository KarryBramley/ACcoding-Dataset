# coding: utf-8
# author: lu yf
# create date: 2017/12/29

from __future__ import division

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, normalized_mutual_info_score
import json
import numpy as np
import warnings
from functools import cmp_to_key
import math
import torch.nn.functional as functional
import torch

warnings.filterwarnings('ignore')


class Evaluation:
    def __init__(self, data_set):
        self.entity_problem_name_emb_dict = {}
        self.entity_user_name_emb_dict = {}
        self.data_set = data_set
        np.random.seed(1)

    def load_emb(self, emb_file):
        """
        load embeddings
        :param emb_name:
        :return:
        """
        with open(emb_file, 'r') as ef:
            emb_dict = json.load(ef)
        return emb_dict

    def evaluation(self, emb_dict, eval_data, linear_model, epoch=0):
        self.epoch = epoch
        entity_emb = emb_dict['ent_embeddings.weight']
        cls_metrics = None
        rec_metrics = None
        self.linear_model = linear_model
        with open('../data/{}/node2id.txt'.format(self.data_set),'r') as e2i_file:
            lines = e2i_file.readlines()

        problem_id_name_dict = {}
        user_id_name_dict = {}
        for i in range(1,len(lines)):
            tokens = lines[i].strip().split('\t')
            if lines[i][0] == 'p':
                problem_id_name_dict[tokens[1]] = int(tokens[0].strip('p'))
            if lines[i][0] == 'u':
                user_id_name_dict[tokens[1]] = int(tokens[0].strip('u'))

        for p_id, p_name in problem_id_name_dict.items():
            # 把entity_emb里面的数组都转换成float类型，但是要加一个list()，又是python2和3的区别orz。。。python2的map函数直接返回列表，python3返回一个迭代器。
            p_emb = list(map(lambda x: float(x),entity_emb[int(p_id)]))
            self.entity_problem_name_emb_dict[p_name] = p_emb

        for u_id,u_name in user_id_name_dict.items():
            u_emb = list(map(lambda x: float(x),entity_emb[int(u_id)]))
            self.entity_user_name_emb_dict[u_name] = u_emb

        pid = []
        label = []
        # print(self.entity_problem_name_emb_dict)
        if eval_data == 'test' or 'valid':
            with open('../data/{}/problem_label.txt'.format(self.data_set), 'r') as problem_label_file:
                problem_label_lines = problem_label_file.readlines()
        elif eval_data == 'train':
            with open('../data/{}/problem_label.txt'.format(self.data_set), 'r') as problem_label_file:
                problem_label_lines = problem_label_file.readlines()
        for line in problem_label_lines:
            tokens = line.strip().split('\t')
            pid.append(self.entity_problem_name_emb_dict[int(tokens[0])])
            label.append(int(tokens[1]))
        # self.kmeans_nmi(x_paper, y_paper, k=5)
        cls_metrics = self.classification(pid, label)

        if eval_data == 'test' or eval_data == 'valid':
            users, problems, rates = set(), set(), {}
            with open(f'../data/{self.data_set}/recommendation_{eval_data}.txt', 'r', encoding='UTF-8') as fin:
                line = fin.readline()
                while line:
                    user, problem, rate = [int(s) for s in line.strip().split('\t')]
                    if rates.get(user) is None:
                        rates[user] = {}
                    rates[user][problem] = float(rate)
                    users.add(user)
                    problems.add(problem)
                    line = fin.readline()
            self.test_x = users
            self.test_y = problems
            self.test_rate = rates
            rec_metrics = self.top_N(10)
        print('='*54)
        return cls_metrics, rec_metrics

    def kmeans_nmi(self, k):
        # km = KMeans(n_clusters=k)
        # km.fit(x,y)
        # y_pre = km.predict(x)

        # nmi = normalized_mutual_info_score(y, y_pre)
        print('-'*24+'kmeans'+'-'*24)
        # print('NMI: {}'.format(nmi))
        test_x = self.test_x
        x = []
        emb_x = self.entity_user_name_emb_dict
        for t in test_x:
            x.append(emb_x[t])

        km = KMeans(n_clusters=k)
        pre_y = km.fit_predict(x)
        with open('../res/{}/cluster_result_{}.csv'.format(self.data_set, self.epoch), 'w') as fr:
            fr.write('uid, cluster\n')
            for i,t in enumerate(test_x):
                fr.write('{}, {} \n'.format(t, pre_y[i]))


    def classification(self, x, y):
        # x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2,random_state=9)

        # lr = LogisticRegression()
        # lr.fit(x_train,y_train)

        # y_valid_pred = lr.predict(x_valid)
        # micro_f1 = f1_score(y_valid, y_valid_pred,average='micro')
        # macro_f1 = f1_score(y_valid, y_valid_pred,average='macro')

        cls_weight = self.linear_model.weight.data
        cls_bias = self.linear_model.bias.data
        y_norm = functional.linear(torch.Tensor(x), cls_weight, cls_bias)
        y_pred = torch.argmax(y_norm, dim=1)
        # if val_data == 'test':
        micro_f1 = f1_score(y, y_pred,average='micro')
        macro_f1 = f1_score(y, y_pred,average='macro')
        print('-'*20+'classification'+'-'*20)
        print ('Macro-F1: {}; Micro-F1: {}'.format(round(macro_f1, 4), round(micro_f1, 4)))
        return {'Macro-F1': round(macro_f1, 4), 'Micro-F1': round(micro_f1, 4)}

    def top_N(self, top_n):
        test_x = self.test_x
        test_y = self.test_y
        test_rate = self.test_rate
        emb_x = self.entity_user_name_emb_dict
        emb_y = self.entity_problem_name_emb_dict
        recommend_dict = {}
        # 模型得到的emb_x和emb_y矩阵都是按用户和题目id顺序排列的，所以直接emb_x[x]就表示id为x的用户的embedding
        for x in test_x:
            recommend_dict[x] = {}
            for y in test_y:
                if x >= len(emb_x):
                    pre = 0
                else:
                    X = np.array(emb_x[x])
                    if y >= len(emb_y):
                        pre = 0
                    else:
                        Y = np.array(emb_y[y])
                        pre = X.dot(Y.T)
                recommend_dict[x][y] = float(pre)

        precision_list = []
        recall_list = []
        ap_list = []
        ndcg_list = []
        rr_list = []

        for x in test_x:
            # tmp_r就是最终推荐的内容，tmp_t用来评估
            tmp_r = sorted(recommend_dict[x].items(), key=cmp_to_key(lambda x, y: cmp(x[1], y[1])), reverse=True)[
                    0: min(len(recommend_dict[x]), top_n)]
            tmp_t = sorted(test_rate[x].items(), key=cmp_to_key(lambda x, y: cmp(x[1], y[1])), reverse=True)[
                    0:min(len(test_rate[x]), top_n)]

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

        precison = sum(precision_list) / len(precision_list)
        recall = sum(recall_list) / len(recall_list)
        f1 = 2 * precison * recall / (precison + recall)
        mean_ap = sum(ap_list) / len(ap_list)
        mrr = sum(rr_list) / len(rr_list)
        mndcg = sum(ndcg_list) / len(ndcg_list)
        print('-'*20+'recommendation'+'-'*20)
        print(f'F1: {round(f1, 4)}; MAP: {round(mean_ap, 4)}; MRR: {round(mrr, 4)}; MNDCG: {round(mndcg, 4)}')
        return {'F1':round(f1,4), 'MAP': round(mean_ap,4), 'MRR': round(mrr, 4), 'NDGG': round(mndcg, 4)}

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

# if __name__ == '__main__':
#     exp = Evaluation()
#     emb1 = exp.load_emb('../res/dblp/embedding.ap_pt_apt+pc_apc.json')
#     exp.evaluation(emb1)

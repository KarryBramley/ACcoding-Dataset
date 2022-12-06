#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'CLH'

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys

sys.path.append(r"recommendation/BINE")
import numpy as np
from sklearn import preprocessing
from data_utils import DataUtils
from graph_utils import GraphUtils
import random
import math
import os
from functools import cmp_to_key
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold


def init_embedding_vectors(node_u, node_v, node_list_u, node_list_v, args):
    """
    initialize embedding vectors
    :param node_u:
    :param node_v:
    :param node_list_u:
    :param node_list_v:
    :param args:
    :return:
    """
    # user
    for i in node_u:
        vectors = np.random.random([1, args.d])
        help_vectors = np.random.random([1, args.d])
        node_list_u[i] = {}
        node_list_u[i]['embedding_vectors'] = preprocessing.normalize(vectors, norm='l2')
        node_list_u[i]['context_vectors'] = preprocessing.normalize(help_vectors, norm='l2')
    # item
    for i in node_v:
        vectors = np.random.random([1, args.d])
        help_vectors = np.random.random([1, args.d])
        node_list_v[i] = {}
        node_list_v[i]['embedding_vectors'] = preprocessing.normalize(vectors, norm='l2')
        node_list_v[i]['context_vectors'] = preprocessing.normalize(help_vectors, norm='l2')


def walk_generator(gul, args):
    """
    walk generator
    :param gul:
    :param args:
    :return:
    """
    gul.calculate_centrality(args.mode)
    if args.large == 0:
        gul.homogeneous_graph_random_walks(percentage=args.p, maxT=args.maxT, minT=args.minT)
    elif args.large == 1:
        gul.homogeneous_graph_random_walks_for_large_bipartite_graph(percentage=args.p, maxT=args.maxT, minT=args.minT)
    elif args.large == 2:
        gul.homogeneous_graph_random_walks_for_large_bipartite_graph_without_generating(datafile=args.train_data,
                                                                                        percentage=args.p,
                                                                                        maxT=args.maxT, minT=args.minT)
    return gul


def get_context_and_negative_samples(gul, args):
    """
    get context and negative samples offline
    :param gul:
    :param args:
    :return: context_dict_u, neg_dict_u, context_dict_v, neg_dict_v,gul.node_u,gul.node_v
    """
    if args.large == 0:
        neg_dict_u, neg_dict_v = gul.get_negs(args.ns)
        print("negative samples is ok.....")
        context_dict_u, neg_dict_u = gul.get_context_and_negatives(gul.G_u, gul.walks_u, args.ws, args.ns, neg_dict_u)
        context_dict_v, neg_dict_v = gul.get_context_and_negatives(gul.G_v, gul.walks_v, args.ws, args.ns, neg_dict_v)
    else:
        neg_dict_u, neg_dict_v = gul.get_negs(args.ns)
        # print len(gul.walks_u),len(gul.walks_u)
        print("negative samples is ok.....")
        context_dict_u, neg_dict_u = gul.get_context_and_negatives(gul.node_u, gul.walks_u, args.ws, args.ns,
                                                                   neg_dict_u)
        context_dict_v, neg_dict_v = gul.get_context_and_negatives(gul.node_v, gul.walks_v, args.ws, args.ns,
                                                                   neg_dict_v)

    return context_dict_u, neg_dict_u, context_dict_v, neg_dict_v, gul.node_u, gul.node_v


def skip_gram(center, contexts, negs, node_list, lam, pa):
    """
    skip-gram
    :param center:
    :param contexts:
    :param negs:
    :param node_list:
    :param lam:
    :param pa:
    :return:
    """
    loss = 0
    I_z = {contexts: 1}  # indication function
    for node in negs:
        I_z[node] = 0
    V = np.array(node_list[center]['embedding_vectors'])
    update = [[0] * V.size]
    for u in I_z.keys():
        if node_list.get(u) is None:
            pass
        Theta = np.array(node_list[u]['context_vectors'])
        X = float(V.dot(Theta.T))  # if x is a very small negative number, such as -200000, then math.exp(-X) will overflow
        try:
            sigmoid = 1.0 / (1 + (math.exp(-X * 1.0)))
            update += pa * lam * (I_z[u] - sigmoid) * Theta
            node_list[u]['context_vectors'] += pa * lam * (I_z[u] - sigmoid) * V
            loss += pa * (I_z[u] * math.log(sigmoid) + (1 - I_z[u]) * math.log(1 - sigmoid))
        # except OverflowError:
        except:
            # sigmoid = 0.0  # because math.exp(-X) + 1 is float('inf')
            pass
        # print "skip_gram:",
        # print(V,Theta,sigmoid,X,math.exp(-X * 1.0),round(math.exp(-X * 1.0),10))
    return update, loss


def KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, gamma):
    """
    KL-divergenceO1
    :param edge_dict_u:
    :param u:
    :param v:
    :param node_list_u:
    :param node_list_v:
    :param lam:
    :param gamma:
    :return:
    """
    loss = 0
    e_ij = edge_dict_u[u][v]

    update_u = 0
    update_v = 0
    U = np.array(node_list_u[u]['embedding_vectors'])
    V = np.array(node_list_v[v]['embedding_vectors'])
    X = float(U.dot(V.T))

    try:
        sigmoid = 1.0 / (1 + (math.exp(-X * 1.0)))
        update_u += gamma * lam * ((e_ij * (1 - sigmoid)) * 1.0 / math.log(math.e, math.e)) * V
        update_v += gamma * lam * ((e_ij * (1 - sigmoid)) * 1.0 / math.log(math.e, math.e)) * U

        loss += gamma * e_ij * math.log(sigmoid)
    except:
        pass
    # except OverflowError:
        #sigmoid = 0.0  # because math.exp(-X) + 1 is float('inf')

    # print "KL:",
    # print(U,V,sigmoid,X,math.exp(-X * 1.0),round(math.exp(-X * 1.0),10))
    return update_u, update_v, loss


def cross_entropy(v, node_list_v, label_list_v, cls_weight, cls_bias, lam, eta):
    loss = 0
    update_v = 0
    nclass = cls_weight.shape[1]
    V = np.array(node_list_v[v]['embedding_vectors'])
    label = label_list_v[v]
    if label == -1:
        return update_v, loss
    Z = V.dot(cls_weight) + cls_bias
    Z_norm = np.exp(Z) / np.exp(Z).sum()
    Y = np.zeros([1, nclass])
    Y[0][label] = 1
    softmax_diag = np.diag((Y - Z_norm).flatten())
    update_v += eta * lam * (softmax_diag.dot(cls_weight.T).sum(axis=0))
    cls_weight += (eta * lam * (Y - Z_norm).T * np.tile(V, (nclass, 1))).T
    cls_bias += eta * lam * (Y - Z_norm)
    try:
        loss += eta * math.log(Z_norm[0][label])
    except:
        pass
    return update_v, loss


def top_N(test_u, test_v, test_rate, node_list_u, node_list_v, top_n):
    recommend_dict = {}
    for u in test_u:
        # 在嵌入结果中找到对应的u和v，把它们的点乘结果保存到recommend_dict，用来排序、做比较，取topN来推荐
        recommend_dict[u] = {}
        for v in test_v:
            if node_list_u.get(u) is None:
                pre = 0
            else:
                U = np.array(node_list_u[u]['embedding_vectors'])
                if node_list_v.get(v) is None:
                    pre = 0
                else:
                    V = np.array(node_list_v[v]['embedding_vectors'])
                    # 这里进行点乘
                    pre = U.dot(V.T)[0][0]
            recommend_dict[u][v] = float(pre)

    precision_list = []
    recall_list = []
    ap_list = []
    ndcg_list = []
    rr_list = []

    for u in test_u:
        # 给每个u排序v的相似度，并取topN，还取了测试集本身自己的权重排序结果（用来验证性能）
        # 下面两行本来用的是内置的cmp(x[1],y[1])，x[1]>y[1]时返回1，在python3里面没有了，需要自己写一个，python3里面的operator.ge(x,y)返回的是true和false，好像不认
        tmp_r = sorted(recommend_dict[u].items(), key=cmp_to_key(lambda x, y: cmp(x[1], y[1])), reverse=True)[
                0:min(len(recommend_dict[u]), top_n)]
        tmp_t = sorted(test_rate[u].items(), key=cmp_to_key(lambda x, y: cmp(x[1], y[1])), reverse=True)[
                0:min(len(test_rate[u]), top_n)]

        # 推荐最终取的推荐内容就是tmp_r这个数组，后面都是在验证效果，分别用了F1、NDGG、MAP和MRR
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
    # print(precison, recall)
    f1 = 2 * precison * recall / (precison + recall)
    map = sum(ap_list) / len(ap_list)
    mrr = sum(rr_list) / len(rr_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return f1, map, mrr, mndcg


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


def classification(node_list_pid):
    label = pd.read_csv(os.path.join('./data', 'daily_2019_pid_label.csv'))
    X = []
    for p in label.pid:
        X.append(node_list_pid[p]['embedding_vectors'])
    X = np.squeeze(np.array(X))
    y = label.label
    kf = KFold(3)
    scores = []
    for train_idx, test_idx in kf.split(X):
        model = LogisticRegressionCV(max_iter=500)
        model.fit(X[train_idx], y[train_idx])
        pred_y = model.predict(X[test_idx])
        scores.append((pred_y == y[test_idx]).mean())
    return np.mean(scores)


def classification_new(test_item, test_label, node_list_v, cls_weight, cls_bias):
    pred_list = []
    label_list = []
    for v in test_item:
        label = test_label[v]
        if label == -1:
            continue
        V = node_list_v[v]['embedding_vectors']
        Z = V.dot(cls_weight) + cls_bias
        Z_norm = np.exp(Z) / np.exp(Z).sum()
        pred = np.argmax(Z_norm, axis=1)
        pred_list.append(pred)
        label_list.append(label)
    from sklearn import metrics
    acc = metrics.accuracy_score(label_list, pred_list)
    micro_f1 = metrics.f1_score(label_list, pred_list, average='micro')
    macro_f1 = metrics.f1_score(label_list, pred_list, average='macro')
    return acc, micro_f1, macro_f1


def train_by_sampling(args):
    model_path = os.path.join('./results', args.model_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    alpha, beta, gamma, eta, lam = args.alpha, args.beta, args.gamma, args.eta, args.lam
    print('======== experiment settings =========')
    print(
        'alpha : %0.4f, beta : %0.4f, gamma : %0.4f, lam : %0.4f, p : %0.4f, ws : %d, ns : %d, maxT : % d, minT : %d, max_iter : %d, d : %d' % (
            alpha, beta, gamma, lam, args.p, args.ws, args.ns, args.maxT, args.minT, args.max_iter, args.d))

    print('========== processing data ===========')
    dul = DataUtils(model_path)
    test_user, test_item, test_rate = dul.read_data(args.test_data)
    valid_user, valid_item, valid_rate = dul.read_data(args.valid_data)
    train_cls_item, test_cls_item, train_label, test_label = None, None, None, None
    if args.cls:
        test_cls_item, test_label = dul.read_label(args.test_label)
        train_cls_item, train_label = dul.read_label(args.train_label)

    print("constructing graph....")
    gul = GraphUtils(model_path)
    gul.construct_training_graph(args.train_data)
    edge_dict_u = gul.edge_dict_u
    edge_list = gul.edge_list
    walk_generator(gul, args)
    print("getting context and negative samples....")
    context_dict_u, neg_dict_u, context_dict_v, neg_dict_v, node_u, node_v = get_context_and_negative_samples(gul, args)

    node_list_u, node_list_v = {}, {}
    init_embedding_vectors(node_u, node_v, node_list_u, node_list_v, args)
    cls_weight = np.random.random([args.d, args.nclass])
    cls_weight = preprocessing.normalize(cls_weight, norm='l2')
    cls_bias = np.random.random([1, args.nclass])
    cls_bias = preprocessing.normalize(cls_bias, norm='l2')
    last_loss, count, epsilon = 0, 0, 1e-3

    print("============== training ==============")
    best_epoch, best_score = 0, 0
    best_node_list_u, best_node_list_v = node_list_u, node_list_v
    for iter in range(0, args.max_iter):
        # s1 = "\r[%s%s]%0.2f%%" % ("*" * iter, " " * (args.max_iter - iter), iter * 100.0 / (args.max_iter - 1))
        # o1_loss: first-order KL divergence loss
        # o2_loss: user-user second-order skip-gram loss
        # o3_loss: question-question second-order skip-gram loss
        # o4_loss: question-knowledge point classification loss
        o1_loss, o2_loss, o3_loss, o4_loss, loss = 0, 0, 0, 0, 0
        visited_u = dict(zip(node_list_u.keys(), [0] * len(node_list_u.keys())))
        visited_v = dict(zip(node_list_v.keys(), [0] * len(node_list_v.keys())))
        random.shuffle(edge_list)
        for i in range(len(edge_list)):
            u, v, w = edge_list[i]
            if not context_dict_u.get(u) or not context_dict_v.get(v):
                continue

            length = len(context_dict_u[u])
            random.shuffle(context_dict_u[u])
            if visited_u.get(u) < length:
                # print(u)
                index_list = list(range(visited_u.get(u), min(visited_u.get(u) + 1, length)))
                for index in index_list:
                    context_u = context_dict_u[u][index]
                    neg_u = neg_dict_u[u][index]
                    # center,context,neg,node_list,eta
                    for z in context_u:
                        tmp_z, tmp_loss = skip_gram(u, z, neg_u, node_list_u, lam, alpha)
                        node_list_u[u]['embedding_vectors'] += tmp_z
                        o2_loss += tmp_loss
                visited_u[u] = index_list[-1] + 3

            length = len(context_dict_v[v])
            random.shuffle(context_dict_v[v])
            if visited_v.get(v) < length:
                # print(v)
                index_list = list(range(visited_v.get(v), min(visited_v.get(v) + 1, length)))
                for index in index_list:
                    context_v = context_dict_v[v][index]
                    neg_v = neg_dict_v[v][index]
                    # center,context,neg,node_list,eta
                    for z in context_v:
                        tmp_z, tmp_loss = skip_gram(v, z, neg_v, node_list_v, lam, beta)
                        node_list_v[v]['embedding_vectors'] += tmp_z
                        o3_loss += tmp_loss
                visited_v[v] = index_list[-1] + 3

            update_u, update_v, tmp_loss = KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, gamma)
            o1_loss += tmp_loss
            node_list_u[u]['embedding_vectors'] += update_u
            node_list_v[v]['embedding_vectors'] += update_v

            if train_cls_item is not None and v in train_cls_item:
                update_v, tmp_loss = cross_entropy(v, node_list_v, train_label, cls_weight, cls_bias, lam, eta)
                o4_loss += tmp_loss
                node_list_v[v]['embedding_vectors'] += update_v

        # for v in train_cls_item:
        #     update_v, tmp_loss = cross_entropy(v, node_list_v, train_label, cls_weight, cls_bias, lam, eta)
        #     # print("old_loss:", loss, 'classification_loss:', tmp_loss)
        #     loss += tmp_loss
        #     node_list_v[v]['embedding_vectors'] += update_v
        loss = o1_loss + o2_loss + o3_loss
        if args.cls:
            loss = -(loss + o4_loss)
            print('Epoch:{}, o1_loss:{}, o2_loss:{}, o3_loss:{}, o4_loss:{}, loss: {}'.format(iter, -o1_loss, -o2_loss, -o3_loss, -o4_loss, loss))
        else:
            loss = -loss
            print('Epoch:{}, o1_loss:{}, o2_loss:{}, o3_loss:{}, loss: {}'.format(iter, -o1_loss, -o2_loss, -o3_loss, loss))
        delta_loss = abs(loss - last_loss)
        if last_loss > loss:
            lam *= 1.05
        else:
            lam *= 0.95
        last_loss = loss

        if iter % args.check_every == 0:
            print("=" * 20, 'valid', '=' * 20)
            if args.rec:
                print("--------------- recommendation ---------------")
                f1, map, mrr, mndcg = top_N(valid_user, valid_item, valid_rate, node_list_u, node_list_v, args.top_n)
                print('F1 : %0.4f, MAP : %0.4f, MRR : %0.4f, NDCG : %0.4f' % (
                    round(f1, 4), round(map, 4), round(mrr, 4), round(mndcg, 4)))
                if mrr > best_score:
                    best_score = mrr
                    best_epoch = iter
                    best_node_list_u = node_list_u
                    best_node_list_v = node_list_v
                    save_to_file(node_list_u, node_list_v, model_path, args)

            if args.cls:
                print("--------------- classification ---------------")
                acc, micro_f1, macro_f1 = classification_new(train_cls_item, train_label, node_list_v, cls_weight,
                                                             cls_bias)
                print('ACC: %0.4f, Micro_F1: %0.4f, Macro_F1: %0.4f' % (
                round(acc, 4), round(micro_f1, 4), round(macro_f1, 4)))
            print('=' * 50)

        if delta_loss < epsilon:
            break
        # sys.stdout.write(s1)
        # sys.stdout.flush()
    save_to_file(best_node_list_u, best_node_list_v, model_path, args)

    print("="*20, 'TESTING', '='*20)
    print(f'The best epoch is {best_epoch}')
    if args.rec:
        print("-------------- recommendation ---------------")
        f1, map, mrr, mndcg = top_N(test_user, test_item, test_rate, best_node_list_u, best_node_list_v, args.top_n)
        print('F1 : %0.4f, MAP : %0.4f, MRR : %0.4f, NDCG : %0.4f' % (
            round(f1, 4), round(map, 4), round(mrr, 4), round(mndcg, 4)))
    if args.cls:
        print("--------------- classification ---------------")
        acc, micro_f1, macro_f1 = classification_new(test_cls_item, test_label, best_node_list_v, cls_weight, cls_bias)
        print('ACC: %0.4f, Micro_F1: %0.4f, Macro_F1: %0.4f' % (round(acc, 4), round(micro_f1, 4), round(macro_f1, 4)))
    print('='*50)
    # if args.lip:
    #     print("============== testing link prediction ===============")
    #     auc_roc, auc_pr = link_prediction(args)
    #     print('link prediction metrics: AUC_ROC : %0.4f, AUC_PR : %0.4f' % (round(auc_roc, 4), round(auc_pr, 4)))


def ndarray_tostring(array):
    string = ""
    for item in array[0]:
        string += str(item).strip() + " "
    return string + "\n"


def save_to_file(node_list_u, node_list_v, model_path, args):
    with open(os.path.join(model_path, args.vectors_u), "w") as fw_u:
        for u in node_list_u.keys():
            fw_u.write(u + " " + ndarray_tostring(node_list_u[u]['embedding_vectors']))
    with open(os.path.join(model_path, args.vectors_v), "w") as fw_v:
        for v in node_list_v.keys():
            fw_v.write(v + " " + ndarray_tostring(node_list_v[v]['embedding_vectors']))


def main():
    parser = ArgumentParser("BiNE", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

    parser.add_argument('--train-data', default=r'./data/daily_valid/rating_train.dat',
                        help='Input graph file.')

    parser.add_argument('--test-data', default=r'./data/daily_valid/rating_test.dat')

    parser.add_argument('--valid-data', default=r'./data/daily_valid/rating_valid.dat')

    parser.add_argument('--train-label', default=r'./data/daily_valid/pid_label_test.dat')

    parser.add_argument('--test-label', default=r'./data/daily_valid/pid_label_train.dat')

    parser.add_argument('--model-name', default='daily_valid_mrr',
                        help='name of models.')

    parser.add_argument('--vectors-u', default=r'vectors_u.dat',
                        help="file of embedding vectors of U")

    parser.add_argument('--nclass', default=5, type=int, help='number of question types.')

    parser.add_argument('--vectors-v', default=r'vectors_v.dat',
                        help="file of embedding vectors of V")

    parser.add_argument('--check-every', default=5, type=int,
                        help='number of epoch when evaluate on valid set.')

    parser.add_argument('--ws', default=5, type=int,
                        help='window size.')

    parser.add_argument('--ns', default=4, type=int,
                        help='number of negative samples.')

    parser.add_argument('--d', default=200, type=int,
                        help='embedding size.')

    parser.add_argument('--maxT', default=32, type=int,
                        help='maximal walks per vertex.')

    parser.add_argument('--minT', default=1, type=int,
                        help='minimal walks per vertex.')

    parser.add_argument('--p', default=0.15, type=float,
                        help='walk stopping probability.')

    parser.add_argument('--alpha', default=0.0001, type=float,
                        help='trade-off parameter alpha.')

    parser.add_argument('--beta', default=0.001, type=float,
                        help='trade-off parameter beta.')

    parser.add_argument('--gamma', default=0.01, type=float,
                        help='trade-off parameter gamma.')
    parser.add_argument('--eta', default=0.1, type=float,
                        help='trade-off parameter eta for semi-supervised learning.')

    parser.add_argument('--lam', default=0.01, type=float,
                        help='learning rate lambda.')
    parser.add_argument('--max-iter', default=300, type=int,
                        help='maximal number of iterations.')

    parser.add_argument('--top-n', default=10, type=int,
                        help='recommend top-n items for each user.')

    parser.add_argument('--rec', default=1, type=int,
                        help='calculate the recommendation metrics.')

    parser.add_argument('--cls', default=0, type=int,
                        help='calculate the link prediction metrics.')

    parser.add_argument('--large', default=0, type=int,
                        help='for large bipartite, 1 do not generate homogeneous graph file; 2 do not generate homogeneous graph')

    parser.add_argument('--mode', default='hits', type=str,
                        help='metrics of centrality')

    args = parser.parse_args()
    train_by_sampling(args)


def bine_train(args):
    sys.exit(train_by_sampling(args))


if __name__ == "__main__":
    sys.exit(main())

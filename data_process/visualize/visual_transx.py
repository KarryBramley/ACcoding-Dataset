from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans


method = 'transe'
entities = 'problems'
save_path = f"embds_visual/transx/{method}"
tsne_file = f"./embeddings/transx/new/{entities}_tsne.json"
use_exist_tsne = True

with open(tsne_file, 'r') as tf:
    tsne_json = json.load(tf)

# 读embedding向量
embd_list = json.load(open(f'./embeddings/transx/new/{method}.json', 'r'))
ent_embeddings = embd_list['ent_embeddings.weight']
embds = {id: emb for id, emb in enumerate(ent_embeddings)}

# 读transx的字典，方便找embedding的id对应的题目原始pid，并且生成X的embedding列表
entity_file = open('./embeddings/transx/new/entity2id.txt', 'r')
entity_id = entity_file.readlines()

i2p_dict = {}
i2u_dict = {}
for ei in entity_id[1:]:
    e, i = ei.split()
    if e[0] == 'P':
        i2p_dict[int(i)] = e
    if e[0] == 'U':
        i2u_dict[int(i)] = e

if tsne_json.get(method) and use_exist_tsne:
    print(f'get existed tsne results of {method}')
    X_norm = np.array(tsne_json[method])

else:
    print(f'generating tsne of {method}...')
    if entities == 'problems':
        X = np.array([embds[i] for i in i2p_dict.keys()])
    else:
        X = np.array([embds[i] for i in i2u_dict.keys()])

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    tsne_json[method] = X_norm.tolist()
    # print(tsne_json)
    with open(tsne_file, 'w') as tf:
        json.dump(tsne_json, tf)


def visual_problems():
    # 读problem的属性，主要是difficulty
    problem_file = open('../data/problems.csv', 'r')
    p_difficulty = {s.split(',')[0]: int(s.split(',')[1]) for s in problem_file.readlines()[1:]}

    # 读problem的统计数据，包括总的ac率和被提交的次数
    problem_statistic_df = pd.read_csv('./statistics/problem_statistic.csv')
    p_acrate = {}
    p_subcnt = {}
    p_avg_cnt = {}
    p_ac_user = {}
    p_cnt_user = {}
    for p in problem_statistic_df.values:
        pid, sub_cnt, user_cnt, avg_cnt, ac_rate, ac_user_rate = p
        p_acrate[pid] = int(ac_rate // 0.1)   # 离散化为0~1之间的几个段，方便搞颜色
        p_acrate[pid] = ac_rate
        # p_subcnt[pid] = int(sub_cnt // 600)   # 为提交次数分段
        p_subcnt[pid] = sub_cnt
        p_cnt_user[pid] = user_cnt
        p_avg_cnt[pid] = avg_cnt
        p_ac_user[pid] = ac_user_rate

    # 找对应的特征，从字典里遍历，分别和X中的embedding是一一对应的
    y_diff = []
    y_ac = []
    y_cnt = []
    for i, pid in i2p_dict.items():
        # X.append(embds[int(i)])
        y_diff.append(p_difficulty[pid])
        y_ac.append(p_acrate[pid])
        y_cnt.append(soft_range(p_cnt_user[pid], 1500))

    font = {'weight': 'normal', 'size': 30, 'family': 'Arial'}

    print('drawing difficulties...')
    plt.figure(figsize=(35, 8))
    plt.subplot(131)

    # 自己组合色卡的方法
    top = plt.cm.get_cmap('summer', 128)
    bottom = plt.cm.get_cmap('autumn', 128).reversed()   # reversed()用来翻转色带
    y_colors = np.vstack((top(np.linspace(0.2, 1, 5)),     # linspace参数：start, stop, num
                          bottom(np.linspace(0.1, 0.9, 5))))
    dif_cmp = ListedColormap(y_colors)

    df = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y_diff, cmap=dif_cmp)
    dif_cb = plt.colorbar(df)
    dif_cb.ax.tick_params(labelsize=20)
    plt.xticks([])
    plt.yticks([])
    # plt.title('Difficulties', fontdict=font)

    print('drawing ac_rates...')
    plt.subplot(132)
    ac_cmp = plt.cm.get_cmap('rainbow', 20).reversed()
    ac = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y_ac, cmap=ac_cmp)
    ac_cb = plt.colorbar(ac)
    ac_cb.ax.tick_params(labelsize=20)
    plt.xticks([])
    plt.yticks([])
    # plt.title('AC ratios', fontdict=font)

    print('drawing sub_cnts...')
    plt.subplot(133)
    cnt_cmp = plt.cm.get_cmap('coolwarm', 18)
    cnt = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y_cnt, cmap=cnt_cmp)
    cnt_cb = plt.colorbar(cnt)
    cnt_cb.ax.tick_params(labelsize=20)
    plt.xticks([])
    plt.yticks([])
    # plt.title('Heats', fontdict=font)

    plt.savefig(save_path + '_problems.png')
    plt.show()


def visual_users():
    # 读用户的属性，主要是学院、年级
    user_df = pd.read_csv('../new_data/users.csv')

    college_dict = {c: i for i, c in enumerate(user_df['college'].unique())}
    grade_dict = {g: i for i, g in enumerate(range(2014, 2020))}
    print(college_dict)
    print(grade_dict)

    u_college, u_grade = {}, {}
    for u in user_df.values:
        uid, college, grade = u
        u_college[uid] = college_dict[college]
        u_grade[uid] = grade_dict[grade]

    # 读problem的统计数据，包括总的ac率和被提交的次数
    user_statistic_df = pd.read_csv('./statistics_new/user_statistic.csv')
    u_ac_rate = {}
    u_sub_cnt = {}
    u_pro_cnt = {}

    for u in user_statistic_df.values:
        _, uid, sub_cnt, ac_rate, pro_cnt = u
        u_sub_cnt[uid] = sub_cnt
        u_ac_rate[uid] = ac_rate
        u_pro_cnt[uid] = pro_cnt

    # 找对应的特征，从字典里遍历，分别和X中的embedding是一一对应的
    y_college = []
    y_grade = []
    y_sub = []
    y_ac = []
    y_pro = []
    for i, uid in i2u_dict.items():
        # X.append(embds[int(i)])
        y_college.append(u_college[uid])
        y_grade.append(u_grade[uid])
        y_sub.append(u_sub_cnt[uid])
        y_ac.append(u_ac_rate[uid])
        y_pro.append(soft_range(u_pro_cnt[uid], 350))

    font = {'weight': 'normal', 'size': 15}

    # print('drawing colleges...')
    # plt.figure(figsize=(20, 8))
    # dif_cmp = plt.cm.get_cmap('tab20c', len(college_dict))
    # cl = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y_college, cmap=dif_cmp)
    # plt.colorbar(cl)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('colleges', fontdict=font)

    print('drawing grades...')
    plt.figure(figsize=(10, 8))
    gr_cmp = ListedColormap(["#D62728", "#9E9AC8", "#E377C2", "#74C476", "#FDAE6B", "#6BAED6"])
    gr = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y_grade, cmap=gr_cmp)
    plt.colorbar(gr)
    plt.xticks([])
    plt.yticks([])
    plt.title('grades', fontdict=font)
    plt.savefig(save_path + '_users_grades.png')
    plt.show()

    # 画统计信息，包括做题数和AC率
    plt.figure(figsize=(20, 8))
    print('drawing problem count...')
    plt.subplot(121)
    pc_cmap = plt.cm.get_cmap('coolwarm', 20)
    pc = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y_pro, cmap=pc_cmap)
    plt.colorbar(pc)
    plt.xticks([])
    plt.yticks([])
    plt.title('problem counts', fontdict=font)

    print('drawing ac rates...')
    plt.subplot(122)
    ac_cmap = plt.cm.get_cmap('coolwarm', 10)
    ac = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y_ac, cmap=ac_cmap)
    plt.colorbar(ac)
    plt.xticks([])
    plt.yticks([])
    plt.title('ac rates', fontdict=font)

    plt.savefig(save_path + '_users_stat.png')
    plt.show()


def soft_range(value, threshold):
    if value > threshold:
        return threshold + 0.01 * threshold
    return value


def kmeans(k):
    print('generate kmeans...')
    if entities == 'users':
        i2e_dict = i2u_dict
    else:
        i2e_dict = i2p_dict

    x = np.array([X_norm[i] for i in i2e_dict.keys()])
    km = KMeans(n_clusters=k)
    y_kmeans = km.fit_predict(x)
    print('x:', i2e_dict.values())
    print('kmeans:', y_kmeans)
    user_clusters = pd.DataFrame.from_dict({'uid': list(i2e_dict.values()), 'cluster': y_kmeans})
    print(user_clusters)
    user_clusters.to_csv(f'./statistics_new/用户聚类/{method}_tsne_user_clusters_{k}.csv', index=False)

    # plt.figure(figsize=(10, 8))
    # kms_cmap = plt.cm.get_cmap('tab10', 10)
    # kms = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y_kmeans, cmap=kms_cmap)
    # plt.colorbar(kms)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('kmeans results')
    #
    # plt.savefig(save_path + '_user_kmeans.png')
    # plt.show()
    return


if __name__ == '__main__':
    visual_problems()
    # kmeans(6)

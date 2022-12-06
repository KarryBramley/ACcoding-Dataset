import pandas as pd
import os
import numpy as np
import pickle
from numpy import random
import math

top_5 = [8, 21, 73, 76, 91]
# 把分类测试集里的题目单独摘出来，在构建pk的时候去掉
pl_test = open('data/rhine_data/daily/daily_2019/problem_label_train.txt').readlines()
test_pid_list = []
# for pl in pl_test:
#     pid, label = pl.split()
#     test_pid_list.append(int(pid))
# print(test_pid_list)

class RhineData:
    def __init__(self, input_dir, output_dir, name, use_contest=False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.node2id_dict = {}
        self.relation2id_dict = {}
        self.use_contest = use_contest
        self.name = name

    def load_data(self):
        if self.use_contest:
            con_df = pd.read_csv(os.path.join(self.input_dir, 'submission_contest.csv'))
            daily_df = pd.read_csv(os.path.join(self.input_dir, 'submission_daily.csv'))
            df = con_df.append(daily_df)
        else:
            df = pd.read_csv(os.path.join(self.input_dir, 'submission_daily.csv'))
        k_df = pd.read_csv(os.path.join(self.input_dir, 'knowledge_points.csv'))
        # self.df保存全量数据，后面还要用来生成测试集
        self.df = df.query("(time > '2019-01-01') & (time < '2020-01-01')")
        pk_df = pd.read_csv(os.path.join(self.input_dir, 'problem_knowledge.csv'))
        self.pk = pk_df
        pc_df = pd.read_csv(os.path.join(self.input_dir, 'problem_contest.csv'))

        # 取前70%为训练集，用来查询的字典和用来生成边的矩阵只记录训练集的东西
        self.split = math.ceil(len(self.df) * 0.7)
        up_df = self.df[: self.split]
        self.up_df = up_df

        # 重新从0开始编一波码，要不占得空间巨大，然后记录一下这个对应的字典
        u_id_dict = {}
        u_idx = 0
        for u in random.permutation(up_df['uid'].unique()):
            u_id_dict[u] = u_idx
            u_idx += 1
        self.u_id_dict = u_id_dict
        self.user_list = u_id_dict.values()

        p_id_dict = {}
        p_idx = 0
        for p in random.permutation(up_df['pid'].unique()):
            p_id_dict[p] = p_idx
            p_idx += 1
        self.p_id_dict = p_id_dict
        self.problem_list = p_id_dict.values()

        k_id_dict = {}
        k_idx = 0
        for k in k_df['kid'].unique():
            k_id_dict[k] = k_idx
            k_idx += 1
        self.k_id_dict = k_id_dict
        self.knowledge_list = k_id_dict.values()

        if self.use_contest:
            c_id_dict = {}
            c_idx = 0
            for c in up_df['cid'].unique():
                c_id_dict[c] = c_idx
                c_idx += 1
            self.c_id_dict = c_id_dict
            self.contest_list = c_id_dict.values()

        # 生成几个连接矩阵
        self.up_adj_matrix = np.zeros([u_idx + 1, p_idx + 1])
        for i in up_df.index:
            row = u_id_dict[up_df.loc[i, 'uid']]
            col = p_id_dict[up_df.loc[i, 'pid']]
            self.up_adj_matrix[row][col] = 1
        print(self.up_adj_matrix)

        self.pk_list = []
        self.pk_adj_matrix = np.zeros([p_idx + 1, k_idx + 1])
        for i in pk_df.index:
            row = p_id_dict.get(pk_df.loc[i, 'pid'])
            col = k_id_dict.get(pk_df.loc[i, 'kid'])
            # 只保留top_5的知识点，去掉分类测试集里的题目
            if (row is not None) and (col is not None) and (col in top_5) and (row not in test_pid_list):
                self.pk_adj_matrix[row][col] = 1
                self.pk_list.append((row, col))
        print(self.pk_adj_matrix)

        if self.use_contest:
            self.pc_adj_matrix = np.zeros([p_idx + 1, c_idx + 1])
            for i in pc_df.index:
                row = p_id_dict.get(pc_df.loc[i, 'pid'])
                col = c_id_dict.get(pc_df.loc[i, 'cid'])
                if row is not None and col is not None:
                    self.pc_adj_matrix[row][col] = 1
            print(self.pc_adj_matrix)

        # 生成两个组合关系的矩阵, 这个计算有点耗时，存在了pkl文件里，直接读就vans
        self.upk_adj_matrix = np.matmul(self.up_adj_matrix, self.pk_adj_matrix)
        if self.use_contest:
            self.upc_adj_matrix = np.matmul(self.up_adj_matrix, self.pc_adj_matrix)

    def write_node_and_relation_2_files(self, relation_list):
        print('writing node2id to file...')
        with open(os.path.join(self.output_dir, self.name + 'node2id.txt'), 'w') as n2i_file:
            num_node = len(self.user_list) + len(self.problem_list) + len(self.knowledge_list)
            if self.use_contest:
                num_node += len(self.contest_list)
            n2i_file.write(str(num_node))
            n2i_file.write('\n')
            idx = 0
            for i in self.user_list:
                node = 'u' + str(i)
                n2i_file.write(node + '\t' + str(idx) + '\n')
                self.node2id_dict[node] = idx
                idx += 1

            for i in self.problem_list:
                node = 'p' + str(i)
                n2i_file.write(node + '\t' + str(idx) + '\n')
                self.node2id_dict[node] = idx
                idx += 1

            for i in self.knowledge_list:
                node = 'k' + str(i)
                n2i_file.write(node + '\t' + str(idx) + '\n')
                self.node2id_dict[node] = idx
                idx += 1

            if self.use_contest:
                for i in self.contest_list:
                    node = 'c' + str(i)
                    n2i_file.write(node + '\t' + str(idx) + '\n')
                    self.node2id_dict[node] = idx
                    idx += 1

        print('writing relation2id to file...')
        with open(os.path.join(self.output_dir, self.name + 'relation2id.txt'), 'w') as r2i_file:
            num_relation = len(relation_list)
            r2i_file.write(str(num_relation) + '\n')
            for i, r in enumerate(relation_list):
                r2i_file.write(str(r) + '\t' + str(i) + '\n')
                self.relation2id_dict[r] = i

    def generate_triples(self, adj_matrix, relation_type):
        print ('gernerating triples for relation {}...'.format(relation_type))
        ridx, cidx = np.nonzero(adj_matrix)
        ridx = list(ridx)
        cidx = list(cidx)
        num_triples = len(ridx)
        train_data = open(os.path.join(self.output_dir, self.name + 'train2id_'+relation_type+'.txt'), 'w')

        for i in range(num_triples):
            n1 = self.node2id_dict[relation_type[0] + str(ridx[i])]
            n2 = self.node2id_dict[relation_type[-1] + str(cidx[i])]
            r = self.relation2id_dict[relation_type]
            w = int(adj_matrix[ridx[i]][cidx[i]])
            train_data.write(str(n1) + '\t' + str(n2) + '\t' + str(r) + '\t' + str(w) + '\n')

        train_data.close()

    def merge_triples(self, relations_list, relation_category):
        print('merging triples for {}...'.format(relation_category))
        merged_data = open(os.path.join(self.output_dir, self.name + 'train2id_' + relation_category + '.txt'), 'w+')
        line_num = 0
        content = ''
        for r in relations_list:
            for line in open(os.path.join(self.output_dir, self.name + 'train2id_' + r + '.txt')):
                content += line
                line_num += 1
        merged_data.writelines(str(line_num) + '\n' + content)

    # 给知识点按题目个数排下序，选题目数最多的五个知识点，作为分类的标签，选择这些知识点对应的题目作为分类验证中的题目
    def _sort_kps(self):
        skill_dt = self.pk.groupby(by=['kid'])
        kp_num = {'kid': [], 'cnt': []}
        for s in skill_dt:
            kp_num['kid'].append(s[0])
            kp_num['cnt'].append(len(s[1]))
        df_kp = pd.DataFrame.from_dict(kp_num)
        df_kp = df_kp.sort_values(by=['cnt'], ascending=False)
        return df_kp.iloc[:5, 0].values

    # 为分类生成数据
    def generate_classification(self, update_label=False):
        print('generating pid_label...')
        top5_kps = self._sort_kps()

        # 要保证题目是那5个知识点的
        p_list = self.pk.query("kid in @top5_kps").sort_values(by=['pid'])
        p_count = p_list['pid'].value_counts().to_dict()

        update_dict = {8: 0, 21: 1, 73: 2, 76: 3, 91: 4}

        fl = open(os.path.join(self.output_dir, self.name + 'problem_label.txt'), 'w')
        for p in p_list.values:
            print(p)
            # 去掉有多个知识点的题目，防止影响分类结果，也不好画图
            if p_count[p[0]] > 1:
                continue
            pid = self.p_id_dict.get(p[0])
            label = self.k_id_dict.get(p[1])
            if pid is not None and label is not None:
                if update_label:
                    label = update_dict[label]
                fl.write(str(pid) + '\t' + str(label) + '\n')


    def generate_recommendation_tests(self, set):
        print('generating recommendation test data...')
        if set == 'test':
            test_df = self.df[self.split:]
        elif set == 'valid':
            valid_split = math.ceil(len(self.up_df) * 0.6)
            test_df = self.up_df[valid_split:]
        fr = open(os.path.join(self.output_dir, self.name + f'recommendation_{set}.txt'), 'w')
        for i in test_df.index:
            r = test_df.loc[i, 'cnt']
            u = self.u_id_dict.get(test_df.loc[i, 'uid'])
            p = self.p_id_dict.get(test_df.loc[i, 'pid'])
            if u is not None and p is not None:
                print(r,u,p)
                fr.write(str(u) + '\t' + str(p) + '\t' + str(r) + '\n')


    def get_id_nid_dict(self):
        print(self.u_id_dict)
        with open(os.path.join(self.output_dir, self.name, 'uid_nid_dict.csv'), 'w') as fr:
            for key, value in self.u_id_dict.items():
                fr.write('{},{}\n'.format(key, value))
        with open(os.path.join(self.output_dir, self.name, 'pid_nid_dict.csv'), 'w') as fr:
            for key, value in self.p_id_dict.items():
                fr.write('{},{}\n'.format(key, value))



def split_cls_data():
    pl = open('data/rhine_data/daily/daily_2019_arir/problem_label.txt')
    cls_data = pl.readlines()
    split = math.ceil(len(cls_data) * 0.7)
    cls_test = cls_data[split:]
    cls_train = cls_data[:split]
    test_file = open('data/rhine_data/daily/daily_2019_arir/problem_label_test.txt', 'w')
    train_file = open('data/rhine_data/daily/daily_2019_arir/problem_label_train.txt', 'w')
    for str_test in cls_test:
        test_file.write(str_test)
    for str_train in cls_train:
        train_file.write(str_train)

def update_labels(dataset):
    pl_file = open('data/rhine_data/daily/daily_2019_arir/problem_label{}.txt'.format(dataset), 'r')
    pl_data = pl_file.readlines()
    update_dict = {'8': 0, '21': 1, '73': 2, '76': 3, '91': 4}
    new_file = open('data/rhine_data/daily/daily_2019_arir/problem_label{}_new.txt'.format(dataset), 'w')
    for pl in pl_data:
        pid, label = pl.split()
        new_label = update_dict[label]
        print(pid, label, new_label)
        new_file.write(pid + '\t' + str(new_label) + '\n')



if __name__ == '__main__':
    rd = RhineData('../new_data', 'data/rhine_data/daily/daily_2019_valid', '', use_contest=False)
    rd.load_data()
    # relation_list = ['up', 'pk', 'pc', 'upk', 'upc']
    relation_list = ['up', 'pk', 'upk']
    # relation_list = ['up']
    rd.write_node_and_relation_2_files(relation_list)

    rd.generate_triples(rd.up_adj_matrix, 'up')
    rd.generate_triples(rd.pk_adj_matrix, 'pk')
    # rd.generate_triples(rd.pc_adj_matrix, 'pc')
    rd.generate_triples(rd.upk_adj_matrix, 'upk')
    # rd.generate_triples(rd.upc_adj_matrix, 'upc')

    # rd.merge_triples(['up', 'upc'], 'IRs')
    # rd.merge_triples(['pk', 'pc', 'upk'], 'ARs')
    rd.merge_triples(['up'], 'IRs')
    rd.merge_triples(['pk', 'upk'], 'ARs')
    #
    rd.generate_recommendation_tests('test')
    rd.generate_recommendation_tests('valid')
    rd.generate_classification(update_label=True)
    # rd.student_cluster()
    rd.get_id_nid_dict()
    print()
    # split_cls_data()
    # update_labels('_train')
    # update_labels('_test')
    # update_labels('')


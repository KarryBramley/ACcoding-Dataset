import math

import pandas as pd
import time
import re
import pickle
import random
import numpy as np
import os

TOP5_KPS = {'K538057': 0, 'K538457': 1, 'K538123': 2, 'K538395': 3, 'K538384': 4}

class IgeData:
    def __init__(self, input_dir, output_dir, data_name):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.data_name = data_name
        return

    def readData(self):
        rt = pd.read_csv(os.path.join(self.input_dir, 'results_daily.csv'))
        rt = rt.query("(time > '2019-01-01') & (time < '2020-01-01')")
        data_dict = {'user_id': [], 'problem_id': [], 'result_list': [], 'knowledge': [], 'date': []}
        sk = pd.read_csv(os.path.join(self.input_dir, 'problem_knowledge.csv'))
        df = rt.merge(sk, on='pid')

        kid = -1
        for i, d in enumerate(df.values):
            user_id = df.loc[i, 'uid']
            problem_id = df.loc[i, 'pid']
            detail_list = df.loc[i, 'detail']
            knowledge = df.loc[i, 'kid']
            date = df.loc[i, 'time']

            if i < len(df) - 1 and user_id == df.loc[i + 1, 'uid'] and problem_id == df.loc[i + 1, 'pid']:
                # 如果这个题目不止一个知识点，判断一下当前这个知识点是不是top5的，如果是，且不是那种拥有不止一个top5的知识点的，就赋值
                if kid == -1 and knowledge in TOP5_KPS.keys():
                    kid = TOP5_KPS[knowledge]
                # 如果不在top5里面，或者是top5里面重复的知识点，就都搞成-1.
                else:
                    kid = -1
            else:
                # 对于只有一个知识点，或者最后一个知识点是top5里的，就赋值，剩下的情况保留当前的kid值（防止前面已经赋过值的情况）
                if knowledge in TOP5_KPS.keys():
                    kid = TOP5_KPS[knowledge]
                result_list = []
                for tp in eval(detail_list):
                    result_list.append(tp[-1])
                data_dict['user_id'].append(user_id)
                data_dict['problem_id'].append(problem_id)
                data_dict['result_list'].append(result_list)
                data_dict['knowledge'].append(kid)
                data_dict['date'].append(int(time.mktime(time.strptime(date, "%Y-%m-%d"))))
                kid = -1
        self.data = pd.DataFrame.from_dict(data_dict)
        print(self.data)

    def generate_edges_and_attrs(self):
        data_dict = {'user_id': [], 'problem_id': [], 'result_id': [], 'date': [], 'knowledge': []}
        attrs_list = []
        for i, d in enumerate(self.data.values):
            data_dict['user_id'].append(self.data.loc[i, 'user_id'])
            data_dict['problem_id'].append(self.data.loc[i, 'problem_id'])
            data_dict['result_id'].append(i)
            data_dict['date'].append(self.data.loc[i, 'date'])
            data_dict['knowledge'].append(self.data.loc[i, 'knowledge'])
            # 在attributes表的对应idx位置，放入结果列表
            attrs_list.append(self.data.loc[i, 'result_list'])
            # attr_idx += 1

        df = pd.DataFrame.from_dict(data_dict)
        df['user_id'], uid_unique = pd.factorize(df['user_id'])
        df['problem_id'], pid_unique = pd.factorize(df['problem_id'])
        user_dict = {'uid': [uid for uid in uid_unique]}
        pro_dict = {'pid': [pid for pid in pid_unique]}

        with pd.ExcelWriter(os.path.join(self.output_dir, self.data_name + 'nid_id_dict.xlsx')) as writer:
            pd.DataFrame.from_dict(user_dict).to_excel(writer, sheet_name='user')
            pd.DataFrame.from_dict(pro_dict).to_excel(writer, sheet_name='problem')

        # df.to_csv('data/ige_data/daily_oj_edges_date_with_knowledge2.csv', index=False)
        print(df)
        df.to_csv(os.path.join(self.output_dir, self.data_name + 'edges_date.csv'), index=False)

        np_attrs = np.zeros((len(attrs_list), 50))
        for i, a in enumerate(attrs_list):
            np_attrs[i][: min(len(a), 50)] = a[: min(len(a), 50)]
        print(np_attrs)
        print(len(np_attrs))
        with open(os.path.join(self.output_dir, self.data_name + 'result_sequence.pkl'), 'wb') as pk:
            pickle.dump(np_attrs, pk)


    def generate_pairs(self):
        df = pd.read_csv(os.path.join(self.output_dir, self.data_name + 'train_edges_date.csv'))
        print(df)
        pairs = {'x': [], 'y': []}
        for user in df['user_id'].unique():
            u_df = df.query("user_id == {}".format(user))
            print(u_df)
            u_len = len(u_df)
            idx_count_dict = {i: 0 for i in u_df.index}
            for i, idx_l in enumerate(u_df.index):
                pairs_count = random.randint(u_len // 100, u_len//10+1)
                for p in range(pairs_count):
                    idx_r = random.choice(u_df.index)   # random.choice是从括号里的列表中随机选择一个项
                    if idx_r == idx_l:
                        continue
                    else:
                        idx_count_dict[idx_r] += 1
                        pairs['x'].append([idx_l, idx_r])

        for problem in df['problem_id'].unique():
            p_df = df.query("problem_id == {}".format(problem))
            print(p_df)
            p_len = len(p_df)
            idx_count_dict = {i: 0 for i in p_df.index}
            for i, idx_l in enumerate(p_df.index):
                pairs_count = random.randint(p_len // 100, p_len//10+1)
                if p_len > 500:
                    pairs_count = random.randint(p_len//500, p_len//100)
                for p in range(pairs_count):
                    idx_r = random.choice(p_df.index)
                    if idx_r == idx_l:
                        continue
                    else:
                        idx_count_dict[idx_r] += 1
                        pairs['y'].append([idx_l, idx_r])

        # print(pairs)
        print(len(pairs['x']))
        print(len(pairs['y']))
        with open(os.path.join(self.output_dir, self.data_name + 'pairs.pkl'), 'wb') as pk:
            pickle.dump(pairs, pk)

    def generate_train_test(self, get_train=False, get_test=False, get_valid=False):
        df = pd.read_csv(os.path.join(self.output_dir, self.data_name + 'edges_date.csv'))
        attrs = pickle.load(open(os.path.join(self.output_dir, self.data_name + 'result_sequence.pkl'), 'rb'))

        df = df.sort_values(by=['date'])
        test_split = math.ceil(len(df) * 0.7)
        train = df[:test_split]
        test = df[test_split:]
        valid_split = math.ceil(len(train) * 0.6)
        valid = train[valid_split:]

        if get_train:
            # 训练数据，取前70%，attrs应该也是对应的位置，不能直接按照新的顺序排列，要不就找不到了
            train_attrs = np.ndarray((max(train['result_id'])+1, 50))
            # train_attrs = np.ndarray([attrs[train.loc[i, 'result_id']] for i in train.index])
            for idx in train.index:
                attr_id = train.loc[idx, 'result_id']
                train_attrs[attr_id] = attrs[attr_id]
            print(train_attrs)
            print(train)
            train.to_csv(os.path.join(self.output_dir, 'train_' + self.data_name + 'edges_date.csv'), index=False)
            with open(os.path.join(self.output_dir, 'train_' + self.data_name + 'result_sequence.pkl'), 'wb') as pk:
                pickle.dump(train_attrs, pk)

        user_nodes = train['user_id'].unique()
        problem_nodes = train['problem_id'].unique()

        if get_test:
            # 测试数据，用来做recommendation，内容是用户id，题目id和提交次数
            test_dict = {'x':[], 'y':[], 'rate':[], 'date':[]}
            for idx in test.index:
                if (test.loc[idx, 'user_id'] in user_nodes) and (test.loc[idx, 'problem_id'] in problem_nodes):
                    detail_id = test.loc[idx, 'result_id']
                    # print(idx, test.loc[idx,'user_id'], test.loc[idx, 'problem_id'], attrs[detail_id])
                    test_dict['x'].append(test.loc[idx, 'user_id'])
                    test_dict['y'].append(test.loc[idx, 'problem_id'])
                    test_dict['rate'].append(np.count_nonzero(attrs[detail_id]))
                    test_dict['date'].append(test.loc[idx, 'date'])
            test_df = pd.DataFrame.from_dict(test_dict)
            test_df.to_csv(os.path.join(self.output_dir, self.data_name + 'test_recommendation.csv'), index=False)

        if get_valid:
            # 验证数据，也用来做recommendation，内容是用户id，题目id和提交次数，和上面是完全一样的，不想写了直接复制
            valid_dict = {'x': [], 'y': [], 'rate': [], 'date': []}
            for idx in valid.index:
                if (valid.loc[idx, 'user_id'] in user_nodes) and (valid.loc[idx, 'problem_id'] in problem_nodes):
                    detail_id = valid.loc[idx, 'result_id']
                    # print(idx, test.loc[idx,'user_id'], test.loc[idx, 'problem_id'], attrs[detail_id])
                    valid_dict['x'].append(valid.loc[idx, 'user_id'])
                    valid_dict['y'].append(valid.loc[idx, 'problem_id'])
                    valid_dict['rate'].append(np.count_nonzero(attrs[detail_id]))
                    valid_dict['date'].append(valid.loc[idx, 'date'])
            valid_df = pd.DataFrame.from_dict(valid_dict)
            valid_df.to_csv(os.path.join(self.output_dir, self.data_name + 'valid_recommendation.csv'), index=False)

    def get_classification(self):
        # 用于分类实验的数据，pid和label，label是知识点
        test_dic = {'pid': [], 'label': []}
        train_df = pd.read_csv(os.path.join(self.output_dir, self.data_name + 'train_edges_date.csv'))

        pid_list = train_df.groupby(by=['problem_id'])
        for p in pid_list:
            pid = p[0]
            kid = p[1].values[0][4]
            if kid != -1:
                test_dic['pid'].append(pid)
                test_dic['label'].append(kid)

        test_df = pd.DataFrame.from_dict(test_dic)
        test_df.to_csv(os.path.join(self.output_dir, self.data_name + 'pid_label.csv'), index=False)

    # 拆分一下pid_label数据，并且把train_edges_date里面对应的分类测试数据label改成-1
    # 2020-12-29，给论文增加分类结果时改
    def split_cls_data(self):
        cls_data = pd.read_csv(os.path.join(self.output_dir, 'pid_label.csv'))
        train_df = pd.read_csv(os.path.join(self.output_dir, 'train_edges_date.csv'))
        split = math.ceil(len(cls_data) * 0.7)
        cls_train = cls_data[: split]
        cls_test = cls_data[split:]

        test_pid_list = list(cls_test.pid)
        train_pid_list = list(cls_train.pid)

        for tr in train_df.values:
            if tr[1] in train_pid_list:
                tr[-1] = -1
        train_df.to_csv(os.path.join(self.output_dir, 'train_edges_date.csv'), index=False)
        cls_train.to_csv(os.path.join(self.output_dir, 'pid_label_train.csv'), index=False)
        cls_test.to_csv(os.path.join(self.output_dir, 'pid_label_test.csv'), index=False)


def save_file(data, file):
    with open(file, 'w') as fw:
        for d in data.values:
            fw.write(d[0] + '\t' + d[1] + '\t' + str(d[2]) + '\n')


# 给知识点按题目个数排下序，选题目数最多的五个知识点，作为分类的标签，选择这些知识点对应的题目作为分类验证中的题目
def sort_kps(sk):
    skill_dt = sk.groupby(by=['kid'])
    kp_num = {'kid': [], 'cnt': []}
    for s in skill_dt:
        kp_num['kid'].append(s[0])
        kp_num['cnt'].append(len(s[1]))
    df_kp = pd.DataFrame.from_dict(kp_num)
    df_kp = df_kp.sort_values(by=['cnt'], ascending=False)
    return df_kp.iloc[:5, 0].values


if __name__ == '__main__':

    ig = IgeData(input_dir='../new_data', output_dir='data/ige_data/daily2019_valid', data_name='')

    # ig.readData()
    # ig.generate_edges_and_attrs()
    # ig.generate_train_test(get_train=True, get_test=True, get_valid=True)
    ig.generate_pairs()
    # ig.get_classification()

    # ig.split_cls_data()

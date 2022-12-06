import pandas as pd
import time
import re
import math
import os

TOP5_KPS = {'K538057': 0, 'K538457': 1, 'K538123': 2, 'K538395': 3, 'K538384': 4}

class BineData():
    def __init__(self, name):
        self.name = name
        return

    def readData(self):
        rt = pd.read_csv('../new_data/submissions_daily.csv')
        rt = rt.query("(time > '2019-01-01') & (time < '2020-01-01')")
        self.sk = pd.read_csv('../new_data/problem_knowledge.csv')

        df = rt.merge(self.sk, on='pid')
        knowledge_list = []
        data_dict = {'user': [], 'problem': [], 'knowledge': [], 'count': [], 'date':[]}

        kid = -1
        for i, d in enumerate(df.values):

            user_id = df.loc[i, 'uid']
            problem_id = df.loc[i, 'pid']
            knowledge = df.loc[i, 'kid']
            date = df.loc[i, 'time']
            count = df.loc[i, 'cnt']

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
                data_dict['user'].append(user_id)
                data_dict['problem'].append(problem_id)
                data_dict['knowledge'].append(kid)
                data_dict['date'].append(int(time.mktime(time.strptime(date, "%Y-%m-%d"))))
                data_dict['count'].append(count)
                kid = -1
        self.data = pd.DataFrame.from_dict(data_dict)
        return self.data

    def problemWithWeight(self, data):
        train_dict = {'user': [], 'problem': [], 'count': []}
        data = data.sort_values(by=['date'])

        test_split = math.ceil(len(data) * 0.7)
        train_df = data[: test_split]
        test_df = data[test_split:]
        valid_split = math.ceil(len(train_df) * 0.6)
        valid_df = train_df[valid_split:]
        self.train_df = train_df

        train_file = f'data/bine_data/{self.name}/rating_train.dat'
        valid_file = f'data/bine_data/{self.name}/rating_valid.dat'
        save_file(train_df, train_file)
        save_file(valid_df, valid_file)

        user_nodes = train_df['user'].unique()
        problem_nodes = train_df.sort_values(by=['problem'])['problem'].unique()

        # test集需要判断一下，所以单独写一下
        with open(f'data/bine_data/{self.name}/rating_test.dat', 'w') as fw:
            for idx in test_df.index:
                user = test_df.loc[idx, 'user']
                problem = test_df.loc[idx, 'problem']
                count = test_df.loc[idx, 'count']
                if user in user_nodes and problem in problem_nodes:
                    fw.write(str.replace(user, 'U', 'u') + '\t' + str.replace(problem, 'P', 'i') + '\t' + str(count) + '\n')
        # save_file(self.test_df, 'data/bine_data/{}_rating_test.dat'.format(self.name))

    # 给知识点按题目个数排下序，选题目数最多的五个知识点，作为分类的标签，选择这些知识点对应的题目作为分类验证中的题目
    def sort_kps(self):
        skill_dt = self.sk.groupby(by=['kid'])
        kp_num = {'kid': [], 'cnt': []}
        for s in skill_dt:
            kp_num['kid'].append(s[0])
            kp_num['cnt'].append(len(s[1]))
        df_kp = pd.DataFrame.from_dict(kp_num)
        df_kp = df_kp.sort_values(by=['cnt'], ascending=False)
        return df_kp.iloc[:5, 0].values

    # 为分类生成数据
    def generate_classification(self):
        print('generating pid_label...')

        # 要保证题目在训练集里
        trained_p = self.train_df['problem'].unique()
        # p_list = self.sk.query("kid in @top5_kps").sort_values(by=['pid'])
        p_list = self.sk.sort_values(by=['pid'])
        p_count = p_list['pid'].value_counts().to_dict()
        pid_label_dict = {'pid':[], 'label':[]}
        for i, p in enumerate(p_list.values):
            # 只保留训练集里面出现过的题目
            if p[0] in trained_p:
                pid_label_dict['pid'].append(str.replace(p[0], 'P', 'i'))
                # 去掉有多个知识点的题目，防止影响分类结果，也不好画图
                if p_count[p[0]] > 1:
                    pid_label_dict['label'].append(-1)
                elif p[1] in TOP5_KPS.keys():
                    pid_label_dict['label'].append(TOP5_KPS[p[1]])
                else:
                    pid_label_dict['label'].append(-1)
        pid_label = pd.DataFrame.from_dict(pid_label_dict)
        # pid_label['label'], _ = pd.factorize(pid_label['label'])
        with open('data/bine_data/{}/pid_label.dat'.format(self.name), 'w') as fw:
            for d in pid_label.values:
                fw.write(str(d[0]) + '\t' + str(d[1]) + '\n')


def split_cls_data():
    pl = open('data/bine_data/daily_valid/pid_label.dat', 'r')
    cls_data = pl.readlines()
    print(cls_data)
    split = math.ceil(len(cls_data) * 0.7)
    cls_train = cls_data[: split]
    cls_test = cls_data[split:]
    print(cls_train)
    print(cls_test)
    train_file = open('data/bine_data/daily_valid/pid_label_train.dat', 'w')
    test_file = open('data/bine_data/daily_valid/pid_label_test.dat', 'w')
    for str_train in cls_train:
        train_file.write(str_train)
    for str_test in cls_test:
        test_file.write(str_test)


def save_file(data, file):
    with open(file, 'w') as fw:
        for idx in data.index:
            user = data.loc[idx, 'user']
            problem = data.loc[idx, 'problem']
            count = data.loc[idx, 'count']
            fw.write(str.replace(user,'U','u') + '\t' + str.replace(problem,'P','i') + '\t' + str(count) + '\n')



if __name__ == '__main__':
    bd = BineData('daily_valid')
    data = bd.readData()
    bd.problemWithWeight(data)
    bd.generate_classification()
    split_cls_data()
    bd.problemWithWeight(data)

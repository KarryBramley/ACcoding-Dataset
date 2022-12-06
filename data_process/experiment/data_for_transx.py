import pandas as pd
import os
from numpy import random
import math
import time
import datetime

class TransxData:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def read_data(self):
        # con_df = pd.read_csv(os.path.join(self.input_dir, 'results_contest.csv'))
        daily_df = pd.read_csv(os.path.join(self.input_dir, 'results_daily.csv'))
        # df = con_df.append(daily_df).sort_values(by=['time', 'uid'])
        df = daily_df.sort_values(by=['time', 'uid'])
        self.sub_df = df.query("(time > '2019-01-01') & (time < '2020-01-01')")
        for sub in self.sub_df.values:
            print(sub[:3])
            time = sub[0]

        self.k_df = pd.read_csv(os.path.join(self.input_dir, 'knowledge_points.csv'))
        self.pk_df = pd.read_csv(os.path.join(self.input_dir, 'problem_knowledge.csv'))
        self.pc_df = pd.read_csv(os.path.join(self.input_dir, 'problem_contest.csv'))

    def generate_entities(self):
        # 给实体编码
        idx = 0
        entity2id_dict = {}
        # 用户
        # np.random.permutation是在原来的顺序上再打乱一下
        for u in random.permutation(self.sub_df['uid'].unique()):
            entity2id_dict[u] = idx
            idx += 1

        # 题目
        for p in random.permutation(self.sub_df['pid'].unique()):
            entity2id_dict[p] = idx
            idx += 1

        # 知识点
        for k in self.k_df['kid'].unique():
            entity2id_dict[k] = idx
            idx += 1

        # 比赛
        # for c in self.sub_df['cid'].unique():
        #     if str(c) != 'nan':
        #         entity2id_dict[c] = idx
        #         idx += 1
        self.entity2id_dict = entity2id_dict
        self.entity_list = entity2id_dict.values()

        entity_num = len(self.entity_list)

        entity2id_file = os.path.join(self.output_dir, 'entity2id.txt')
        fe = open(entity2id_file, 'w')
        fe.write(str(entity_num) + '\n')
        for entity, id in entity2id_dict.items():
            print(entity, id)
            fe.write(entity + '\t' + str(id) + '\n')

    def generate_relations(self):
        # 找一下关系，给它们起名字并且编码
        relation2id_dict = {}
        idx = 0
        triple_list = []

        # 因为题目相关的是静态的关系，所以就放到前面，之后split的时候让测试集和验证集都是交互数据好像比较合理

        # 题目-知识点的包含关系 contain/unique(multiple)
        problem_group = self.pk_df.groupby('pid')
        rel_contain_root = 'contain/'
        relation2id_dict[rel_contain_root+'unique'] = idx
        relation2id_dict[rel_contain_root+'multiple'] = idx + 1
        idx += 2
        for problem, pk_relation in problem_group:
            kp_num = len(pk_relation)
            if kp_num > 1:
                relation = rel_contain_root + 'multiple'
            else:
                relation = rel_contain_root + 'unique'
            for pk in pk_relation.values:
                knowledge = pk[1]
                if problem in self.entity2id_dict.keys():
                    triple = str(self.entity2id_dict[problem]) + '\t' \
                             + str(self.entity2id_dict[knowledge]) + '\t' \
                             + str(relation2id_dict[relation])
                    triple_list.append(triple)

        # 题目-比赛的所属关系 belong_to/order_0,1,2...
        # rel_belong_root = 'belong_to/'
        # for pc in self.pc_df.values:
        #     print(pc)
        #     contest = pc[0]
        #     problem = pc[1]
        #     order = int(pc[2])
        #     relation = rel_belong_root + 'order_' + str(order)
        #     if relation not in relation2id_dict.keys():
        #         relation2id_dict[relation] = idx
        #         idx += 1
        #     if problem in self.entity2id_dict.keys() and contest in self.entity2id_dict.keys():
        #         triple = str(self.entity2id_dict[problem]) + '\t' \
        #                  + str(self.entity2id_dict[contest]) + '\t' \
        #                  + str(relation2id_dict[relation])
        #         triple_list.append(triple)

        # 用户-题目的提交关系 分日常和比赛，submit/daily(或contest)/result_AC...
        rel_submit_root = 'submit/'
        for d in self.sub_df.values:
            print(d)
            # contest = d[0]
            # user = d[2]
            # problem = d[3]
            # detail_list = d[4]
            user = d[1]
            problem = d[2]
            detail_list = d[3]
            rel_submit = rel_submit_root + 'daily/'

            # 先生成关系
            # if str(contest) == 'nan':
            #     rel_submit = rel_submit_root + 'daily/'
            # else:
            #     rel_submit = rel_submit_root + 'contest/'
            # detail对contest和daily来说都是一样的，统一处理就行
            for detail in eval(detail_list):
                result = detail[-1]
                # 生成完整的关系，保存到关系dict里面去
                relation = rel_submit + 'result_' + str(result)
                if relation not in relation2id_dict.keys():
                    relation2id_dict[relation] = idx
                    idx += 1
                # 生成三元组，都是用的id
                triple = str(self.entity2id_dict[user]) + '\t' \
                         + str(self.entity2id_dict[problem]) + '\t' \
                         + str(relation2id_dict[relation])
                triple_list.append(triple)

        relation_file = os.path.join(self.output_dir, 'relation2id.txt')
        fr = open(relation_file, 'w')
        relation_num = len(relation2id_dict)
        fr.write(str(relation_num) + '\n')
        for relation, id in relation2id_dict.items():
            fr.write(relation + '\t' + str(id) + '\n')

        self.write_triples(triple_list, 'triples_all.txt')

    def split_data(self):
        triples_file = os.path.join(self.output_dir, 'triples_all.txt')
        ft = open(triples_file, 'r')
        triple_num = int(ft.readline())
        triples = ft.readlines()[1:]
        train_split = math.ceil(triple_num * 0.7)
        train_triples = triples[: train_split]
        test_triples = triples[train_split: ]
        # valid_split = math.ceil(triple_num * 0.8)
        self.write_triples(triples[: train_split], 'train2id.txt')
        # self.write_triples(triples[train_split: valid_split], 'valid2id.txt')

        users, problems = set(), set()
        for tr in train_triples:
            u, p, r = tr.split()
            users.add(u)
            problems.add(p)

        test_triples_cleaned = []
        for tr in test_triples:
            u, p, r = tr.split()
            if u in users and p in problems:
                test_triples_cleaned.append(tr)
        self.write_triples(test_triples_cleaned, 'test2id.txt')

    def split_valid(self):
        train_file = os.path.join(self.output_dir, 'train2id.txt')
        ft = open(train_file, 'r')
        train_num = int(ft.readline())
        train_triples = ft.readlines()[1:]
        valid_split = math.ceil(train_num * 0.6)
        self.write_triples(train_triples[valid_split:], 'valid.txt')

    def write_triples(self, triples, file):
        fw = open(os.path.join(self.output_dir, file), 'w')
        triple_num = len(triples)
        fw.write(str(triple_num) + '\n')
        for tr in triples:
            fw.write(tr.rstrip() + '\n')

    def problem_label(self):
        TOP5_KPS = ['K538057', 'K538457', 'K538123', 'K538395', 'K538384']
        topkpid = {}
        idx = 0
        for kp in TOP5_KPS:
            topkpid[str(self.entity2id_dict[kp])] = idx
            idx += 1
        ft = open(os.path.join(self.output_dir, 'triples_all.txt'), 'r')
        triples = ft.readlines()
        pid_label_dict = {}
        print(topkpid)
        for tr in triples[1:]:
            head, tail, relation = tr.split()
            if relation in ['0', '1']:
                if tail in ['7601', '7684', '7614', '7669', '7666'] and (head not in pid_label_dict.keys()):
                    pid_label_dict[head] = topkpid[tail]
        fl = open(os.path.join(self.output_dir, 'pid_label.txt'), 'w')
        print(pid_label_dict)
        for pid, label in pid_label_dict.items():
            fl.write(pid + '\t' + str(label) + '\n')

if __name__ == '__main__':
    td = TransxData('../new_data', './data/transx_data/2019_daily')
    # td.read_data()
    # td.generate_entities()
    # td.generate_relations()
    # td.split_data()
    # td.split_valid()
    # td.problem_label()

    # 处理一下数据
    d_df = pd.read_csv('../new_data/submissions_daily.csv')
    for i in d_df.index:
        date = d_df.loc[i, 'time']
        fd = time.strptime(date, "%Y-%m-%d")
        new_fd = time.strftime("%Y-%m-%d", fd)
        print(new_fd)
        d_df.loc[i, 'time'] = new_fd
    print(d_df)
    d_df.to_csv('../new_data/submissions_daily_new.csv', index=False)

    c_df = pd.read_csv('../new_data/submissions_contest.csv')
    for i in c_df.index:
        date = c_df.loc[i, 'time']
        fd = time.strptime(date, "%Y-%m-%d")
        new_fd = time.strftime("%Y-%m-%d", fd)
        print(new_fd)
        c_df.loc[i, 'time'] = new_fd
    print(c_df)
    c_df.to_csv('../new_data/submissions_contest_new.csv', index=False)

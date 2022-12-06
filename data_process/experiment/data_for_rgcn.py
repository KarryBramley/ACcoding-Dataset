import pandas as pd
import numpy as np
import os
import math
import time

class RgcnData:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def read_submission_data(self):
        daily_df = pd.read_csv(os.path.join(self.input_dir, 'submissions_daily.csv'))
        self.sub_df = daily_df.query("(time > '2019-01-01') & (time < '2020-01-01')")
        # self.sub_df = df.sort_values(by=['time'])

    def read_feedback_data(self):
        daily_df = pd.read_csv(os.path.join(self.input_dir, 'results_daily.csv'))
        #self.feed_df = daily_df.sort_values(by=['time'])
        self.feed_df = daily_df.query("(time > '2019-01-01') & (time < '2020-01-01')")

    def generate_submission_triples(self):
        with open(os.path.join(self.output_dir, 'triples.txt'), 'w') as tr_file:
            for d in self.sub_df.values:
                print(d)
                # time = d[0]
                uid = d[1]
                pid = d[2]
                cnt = d[3]
                correct = d[4]
                relation = 'AC' if correct == 1 else 'non-AC'
                relation += '/' + str(cnt)
                tr_file.write(uid + '\t' + relation + '\t' + pid + '\n')

    def generate_feedback_triples(self):
        results_dict = {1: 'AC', 2: 'CE', 3: 'IFNR', 4: 'JG', 5: 'MLE', 6: 'PE', 7: 'REP', 8: 'REG', 9: 'TLE', 10: 'WA',
                        11: 'WT', 12: 'OE'}
        triple_dict = {'uid': [], 'time': [], 'result': [], 'pid': []}
        for d in self.feed_df.values:
            print(d)
            date = d[0]
            uid = d[1]
            pid = d[2]
            detail_list = eval(d[3])
            for detail in detail_list:
                result = results_dict[detail[2]]
                time_str = str(date) + ' ' + detail[0]
                time_time = time.mktime(time.strptime(time_str, '%Y-%m-%d %H:%M'))
                triple_dict['uid'].append(uid)
                triple_dict['time'].append(time_time)
                triple_dict['result'].append(result)
                triple_dict['pid'].append(pid)
        triple_df = pd.DataFrame.from_dict(triple_dict)
        print(triple_df)

        with open(os.path.join(self.output_dir, 'triples.txt'), 'w') as tr_file:
            for d in triple_df.values:
                uid = d[0]
                time_time = d[1]
                result = d[2]
                pid = d[3]
                relation = result + '/' + str(time_time)
                tr_file.write(uid + '\t' + relation + '\t' + pid + '\n')

    def split_data(self):
        triples_file = os.path.join(self.output_dir, 'triples.txt')
        ft = open(triples_file, 'r')
        triples = ft.readlines()
        triple_num = len(triples)
        train_split = math.ceil(triple_num * 0.7)

        train_triples = triples[: train_split]
        test_triples = triples[train_split:]

        print('len train:', len(train_triples))
        print('len test:', len(test_triples))

        users, problems = set(), set()
        for tr in train_triples:
            u, r, p = tr.split()
            users.add(u)
            problems.add(p)
        print(len(users), users)
        print(len(problems), problems)

        test_triples_cleaned = []
        for tr in test_triples:
            u, r, p = tr.split()
            print(u, r, p, end='\t')
            if u in users and p in problems:
                print('yes')
                test_triples_cleaned.append(tr)

        self.write_triples(train_triples, 'train.txt')
        self.write_triples(test_triples_cleaned, 'test.txt')

    def split_valid(self):
        train_file = os.path.join(self.output_dir, 'train.txt')
        ft = open(train_file, 'r')
        train_triples = ft.readlines()
        train_num = len(train_triples)
        valid_split = math.ceil(train_num * 0.6)
        self.write_triples(train_triples[valid_split:], 'valid.txt')

    def write_triples(self, triples, file):
        fw = open(os.path.join(self.output_dir, file), 'w')
        for tr in triples:
            fw.write(tr.rstrip() + '\n')

    def generate_static_triples(self):
        pk_df = pd.read_csv(os.path.join(self.input_dir, 'problem_knowledge.csv'))
        with open(os.path.join(self.output_dir, 'statics.txt'), 'w') as st_file:
            for pk in pk_df.values:
                p, k = pk
                relation = 'contain'
                st_file.write(p + '\t' + relation + '\t' + k + '\n')


if __name__ == '__main__':
    rd = RgcnData('../new_data', './data/rgcn_data/feed_rec/daily_valid')
    # rd.read_feedback_data()
    # rd.generate_feedback_triples()
    # rd.read_submission_data()
    # rd.generate_submission_triples()
    rd.generate_static_triples()
    # rd.split_data()
    # rd.split_valid()




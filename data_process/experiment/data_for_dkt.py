import pandas as pd
import re
import time


class DktData():
    def __init__(self):
        return

    def read_data(self):
        rt = pd.read_csv('../data/results_daily.csv')
        rt = rt.query("(time > '2019-01-01') & (time < '2020-01-01')")
        sk = pd.read_csv('../data/problem_knowledge.csv')
        df = rt.merge(sk, on='pid')
        print(df)
        knowledge_list = []
        data_dict = {'user_id': [], 'problem_id': [], 'knowledge_list': [], 'correct_list': [], 'time': []}

        for i, d in enumerate(df.values):
            user_id = df.loc[i, 'uid']
            problem_id = df.loc[i, 'pid']
            detail = df.loc[i, 'detail']
            knowledge = df.loc[i, 'kid']
            date = df.loc[i, 'time']
            # print(user_id, problem_id, knowledge, detail, date)
            if i < len(df) - 1 and user_id == df.loc[i + 1, 'uid'] and problem_id == df.loc[i + 1, 'pid']:
                knowledge_list.append(knowledge)
            else:
                knowledge_list.append(knowledge)
                detail_list = str.split(detail, '),')
                correct_list = []
                for tp in detail_list:
                    str_split = str.split(tp, ',')
                    c = [int(s) for s in re.findall(r"\d+", str_split[-1])][0]
                    time_str = [date+' '+s for s in re.findall(r"\d\d:\d\d", str_split[0])][0]
                    time_stamp = int(time.mktime(time.strptime(time_str,"%Y-%m-%d %H:%M")))
                    correct_list.append((c, time_stamp))
                correct_list = [(int(s[0] == 1), s[1]) for s in correct_list]  # 转换成0和1两种结果
                data_dict['user_id'].append(user_id)
                data_dict['problem_id'].append(problem_id)
                data_dict['knowledge_list'].append(knowledge_list)
                data_dict['correct_list'].append(correct_list)
                data_dict['time'].append(int(time.mktime(time.strptime(date, "%Y-%m-%d"))))
                knowledge_list = []
        return pd.DataFrame.from_dict(data_dict)


    # 题目作为skill_id，记录重复提交
    def generate_data_1(self, data):
        data_dict = {'user_id':[], 'skill_id':[], 'correct':[], 'time':[]}
        for i, d in enumerate(data.values):
            user_id = data.loc[i, 'user_id']
            problem_id = data.loc[i, 'problem_id']
            correct_list = data.loc[i, 'correct_list']

            for correct, time in correct_list:
                data_dict['user_id'].append(user_id)
                data_dict['skill_id'].append(problem_id)
                data_dict['correct'].append(correct)
                data_dict['time'].append(time)
        df = pd.DataFrame.from_dict(data_dict)
        df = df.sort_values(by=['time'])
        df.to_csv('data/dkt_data/contestData2019_1.csv', index=False)
        print(df)

    def generate_data_2(self, data):
        data_dict = {'user_id': [], 'skill_id': [], 'correct': [], 'time': []}
        for i, d in enumerate(data.values):
            user_id = data.loc[i, 'user_id']
            correct_list = data.loc[i, 'correct_list']

            knowledge_list = data.loc[i, 'knowledge_list']
            for correct, time in correct_list:
                for knowledge in knowledge_list:
                    data_dict['user_id'].append(user_id)
                    data_dict['skill_id'].append(knowledge)
                    data_dict['correct'].append(correct)
                    data_dict['time'].append(time)
        df = pd.DataFrame.from_dict(data_dict)
        df = df.sort_values(by=['user_id', 'time'])
        df.to_csv('dkt_data/contestData2019_2.csv', index=False)
        print(df)


    def generate_data_3(self, data):
        # skill_id-知识点，一个题有多个知识点时，每个知识点都对应一条数据（eg. 题目p1对应3个知识点(k1,k2,k3)，则生成三条数据，skill_id分别等于k1,k2,k3，其他值相同）
        # correct-是否正确（0或1），只记一个结果，即这道题是否做对过，多次尝试时，有一次正确结果就记为1
        data_dict = {'user_id': [], 'skill_id': [], 'correct': [], 'time': []}
        for i, d in enumerate(data.values):
            user_id = data.loc[i, 'user_id']
            correct_list = data.loc[i, 'correct_list']
            knowledge_list = data.loc[i, 'knowledge_list']
            time = data.loc[i, 'time']
            correct = int(sum([c[0] for c in correct_list]) > 0)    # 只取最终结果，如果曾经正确过，就计1
            for knowledge in knowledge_list:
                data_dict['user_id'].append(user_id)
                data_dict['skill_id'].append(knowledge)
                data_dict['correct'].append(correct)
                data_dict['time'].append(time)
        df = pd.DataFrame.from_dict(data_dict)
        df = df.sort_values(by=['time'])
        print(df)
        df.to_csv('data/dkt_data/dailyData2019_3.csv', index=False)

    def generate_data_4(self,data):
        # skill_id-知识点，多个知识点时，只记第一个
        # correct-是否正确，一次提交记一条数据
        data_dict = {'user_id': [], 'skill_id': [], 'correct': [], 'time': []}
        for i, d in enumerate(data.values):
            user_id = data.loc[i, 'user_id']
            correct_list = data.loc[i, 'correct_list']
            knowledge_list = data.loc[i, 'knowledge_list']
            for correct,time in correct_list:
                data_dict['user_id'].append(user_id)
                data_dict['skill_id'].append(knowledge_list[0])
                data_dict['correct'].append(correct)
                data_dict['time'].append(time)
        df = pd.DataFrame.from_dict(data_dict)
        df = df.sort_values(by=['user_id', 'time'])
        df.to_csv('dkt_data/contestData2019_4.csv', index=False)

    def cut_part_of_data(self, month, use_contest=True):
        time_str = "2019-{}-1".format(month)
        if use_contest:
            d = "contest"
        else:
            d = "daily"
        end = int(time.mktime(time.strptime(time_str, "%Y-%m-%d")))
        df = pd.read_csv('data/dkt_data/{}Data2019_3.csv'.format(d))
        part_df = df.query("time < {}".format(end))
        print(part_df)
        print(len(part_df['skill_id'].unique()))
        part_df.to_csv("data/dkt_data/{}_2019_kp_before_{}.csv".format(d, month), index=False)

if __name__ == '__main__':
    dkt = DktData()
    # data = dkt.read_data()
    # print(data)
    # dkt.generate_data_3(data)
    dkt.cut_part_of_data(7, False)


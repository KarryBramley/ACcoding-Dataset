import pandas as pd
import csv
import math
import time


class DkvmnData():
    def __init__(self):
        return

    def read_data(self):
        daily_df = pd.read_csv('../new_data/results_daily.csv').query("(time > '2019-01-01') & (time < '2020-01-01')")
        kp_df = pd.read_csv('../new_data/problem_knowledge.csv')
        daily_df = daily_df.merge(kp_df, on='pid')
        rt_dict = {'user_id': [], 'skill_id': [], 'correct': [], 'time': []}
        for result in daily_df.values:
            date = result[0]
            uid = result[1]
            kid = result[-1]
            detail_list = result[-2]
            for detail in eval(detail_list):
                re = detail[-1]
                print(uid, kid, re, date)
                rt_dict['user_id'].append(uid)
                rt_dict['skill_id'].append(kid)
                rt_dict['correct'].append(re)
                rt_dict['time'].append(date)
        rt = pd.DataFrame.from_dict(rt_dict).sort_values(by='time')
        rt.to_csv('./data/dkt_data/daily_2019_kpre_all.csv', index=False)
        exit(0)
        # rt = pd.read_csv('data/dkt_data/dailyData2019_3.csv')
        # rt = pd.read_csv('data/test_data/assistment_test15.csv')

        time_str = "2019-7-1"
        end = int(time.mktime(time.strptime(time_str, "%Y-%m-%d")))
        # rt = rt.query("time < {}".format(end))

        rt['skill_id'], _ = pd.factorize(rt['skill_id'], sort=True)
        return rt

    def generate_data(self, data, data_name):
        train_writer = csv.writer(open('data/dkvmn_data/{}_train.csv'.format(data_name), 'w', encoding='utf-8', newline=''),
                                  dialect='excel')
        test_writer = csv.writer(open('data/dkvmn_data/{}_test.csv'.format(data_name), 'w', encoding='utf-8', newline=''),
                                 dialect='excel')
        valid_writer = csv.writer(open('data/dkvmn_data/{}_valid.csv'.format(data_name), 'w', encoding='utf-8', newline=''),
                                 dialect='excel')
        users_len = len(data['user_id'].unique())
        group_data = data.groupby('user_id')
        split = math.ceil(len(group_data) * 0.6)
        for i, d in enumerate(group_data):
            batch = d[1]
            batch = batch.sort_values(by=['time'])
            batch = batch.query("skill_id > -1")
            count = len(batch)
            # 如果是长度为1的序列，就去掉
            if count < 2:
                continue
            questions = batch['skill_id'].values
            answers = batch['correct'].values
            if i < split:
                train_writer.writerow([str(count)])
                train_writer.writerow(questions)
                train_writer.writerow(answers)
            elif split <= i < split + math.ceil(split * 0.2):
                valid_writer.writerow([str(count)])
                valid_writer.writerow(questions)
                valid_writer.writerow(answers)
            else:
                test_writer.writerow([str(count)])
                test_writer.writerow(questions)
                test_writer.writerow(answers)



if __name__ == '__main__':
    dd = DkvmnData()
    data = dd.read_data()
    dd.generate_data(data, 'daily2019_kpre_all')
    print(len(data['skill_id'].unique()))

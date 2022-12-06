import sqlConnect
import csv

conn = sqlConnect.sql_connect()

class TransformLists():
    def __init__(self):
        return

    def get_transform_list(self, type):
        if type == 'problem':
            return self.get_problem_list()
        elif type == 'contest':
            return self.get_contest_list()
        elif type == 'user':
            return self.get_user_list()
        elif type == 'code':
            return self.get_code_list()
        else:
            return -1

    # 读取重新编码的problem结果
    def get_problem_list(self):
        print('loading problem id...')
        problem_list = conn.select("""select pid_, pid from n_problems""")
        problem_transform_list = {}
        for pid_, pid in problem_list:
            problem_transform_list[pid] = pid_
        print('problem transform list: ', problem_transform_list)
        return problem_transform_list

    # 读取重新编码的contest结果
    def get_contest_list(self):
        print('loading contest id...')
        contest_list = conn.select("""select cid_, cid from n_contests""")
        contest_transform_list = {}
        for cid_, cid in contest_list:
            contest_transform_list[cid] = cid_
        print('contest transform list: ', contest_transform_list)
        return contest_transform_list

    # 读取重新编码的user结果
    def get_user_list(self):
        print('loading user id...')
        user_list = conn.select("""select uid_, uid from n_users""")
        user_transform_list = {}
        for uid_, uid in user_list:
            user_transform_list[uid] = uid_
        print('user transform list: ', user_transform_list)
        return user_transform_list

    # 读取重新编码的code结果，记录在字典里，便于查询
    def get_code_list(self):
        print('loading code id...')
        csv_reader = csv.reader(open('./data/code_transform.csv', 'r', encoding='utf-8'))
        code_transform_list = {}
        print('reading recoded code...')
        for row in csv_reader:
            code_transform_list[row[0]] = row[1]
        print('code transform list: ', code_transform_list)
        return code_transform_list
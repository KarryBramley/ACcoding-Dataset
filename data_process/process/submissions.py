import sqlConnect as sc
import TransformLists as tl
import csv
import pandas as pd

conn = sc.sql_connect()
transformer = tl.TransformLists()
problem_transform_list = transformer.get_transform_list('problem')
contest_transform_list = transformer.get_transform_list('contest')
user_transform_list = transformer.get_transform_list('user')
#code_transform_list = transformer.get_transform_list('code')


def submissions_contest():
    print('读数据...')
    contest_submissions = conn.select("""SELECT contest_id as cid, DATE(created_at) as time, creator_id as uid, problem_id as pid, COUNT(*) as cnt
                                FROM submissions_contest 
                                WHERE creator_id in (SELECT uid FROM n_users)
                                    AND problem_id in (SELECT pid FROM n_problems)
                                    AND contest_id in (SELECT cid FROM n_contests)
                                GROUP BY creator_id,problem_id,contest_id,DATE(created_at) 
                                ORDER BY contest_id,DATE(created_at),creator_id, problem_id""")
    csv_writer = csv.writer(open('./data/submissions_contest.csv', 'w', encoding='utf-8', newline=''),
                            dialect='excel')
    title = ['cid', 'time', 'uid', 'pid', 'cnt']
    csv_writer.writerow(title)
    for i, sub in enumerate(contest_submissions):
        cid = contest_transform_list[sub[0]]
        time = sub[1]
        uid = user_transform_list[sub[2]]
        pid = problem_transform_list[sub[3]]
        cnt = sub[4]
        row = [cid, time, uid, pid, cnt]
        print(row)
        csv_writer.writerow(row)


def submissions_daily():
    print('读数据...')
    contest_submissions = conn.select("""SELECT DATE(created_at) as time, creator_id as uid, problem_id as pid, COUNT(*) as cnt
                                    FROM submissions_daily 
                                    WHERE creator_id in (SELECT uid FROM n_users)
                                        AND problem_id in (SELECT pid FROM n_problems)
                                        AND (contest_id is NULL OR contest_id in (SELECT cid FROM n_contests))
                                    GROUP BY creator_id,problem_id,DATE(created_at)
                                    ORDER BY DATE(created_at),creator_id, problem_id""")
    csv_writer = csv.writer(open('./data/submissions_daily.csv', 'w', encoding='utf-8', newline=''),
                            dialect='excel')
    title = ['time', 'uid', 'pid', 'cnt']
    csv_writer.writerow(title)
    for i, sub in enumerate(contest_submissions):
        time = sub[0]
        uid = user_transform_list[sub[1]]
        pid = problem_transform_list[sub[2]]
        cnt = sub[3]
        row = [time, uid, pid, cnt]
        print(row)
        csv_writer.writerow(row)


def add_correct_2_submissions(data='daily'):
    submission_df = pd.read_csv('./data/submissions_{}.csv'.format(data))
    result_df = pd.read_csv('./data/results_{}.csv'.format(data))

    for i in submission_df.index:
        detail_list = result_df.loc[i, 'detail']
        correct = 0
        for detail in eval(detail_list):
            if detail[-1] == 1:
                correct = 1
                break
        submission_df.loc[i, 'correct'] = correct
        print(submission_df.loc[i])

    print(submission_df)
    submission_df.to_csv('./data/submission_{}_new.csv'.format(data))


if __name__ == '__main__':
    # add_correct_2_submissions()
    solution_df = pd.read_csv('./data/solutions.csv')
    new_solution_df = solution_df.drop('code', axis=1).fillna(value=0)
    new_solution_df.to_csv('./data/solutions_wo_code.csv', index=False)

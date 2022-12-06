import sqlConnect as sc
import TransformLists as tl
import datetime
import csv

conn = sc.sql_connect()
transformer = tl.TransformLists()
problem_transform_list = transformer.get_transform_list('problem')
user_transform_list = transformer.get_transform_list('user')
code_transform_list = transformer.get_transform_list('code')

beginDay = datetime.date(2018,1,1)
delta = datetime.timedelta(days=1)
endDay = beginDay + delta

#反馈结果，按字母顺序排列，OE除外
resultDic = {'AC':1, 'CE':2, 'IFNR':3, 'JG':4, 'MLE':5, 'PE':6, 'REG':7, 'REP':8, 'TLE':9, 'WA':10, 'WT':11, 'OE':12 }

print('读数据...')
selectStr = """SELECT DATE(created_at) as time, creator_id as uid, problem_id as pid, DATE_FORMAT(created_at,'%H:%i') as clock, result, sc.id 
                FROM submissions_daily s, submission_codes sc
                WHERE s.id = sc.submission_id
                AND s.creator_id in (SELECT uid FROM n_users)
                AND s.problem_id in (SELECT pid FROM n_problems)
                AND (s.contest_id is NULL OR s.contest_id in (SELECT cid FROM n_contests)) 
                ORDER BY time, uid, pid, clock"""
submissions = conn.select(selectStr)

#利用csv包来读文件
#csv_reader=csv.reader(open('./data/results_daily.csv','r',encoding='gb18030'))
#这里要加上encoding=''这个东西，因为文件是gbk编码的
#在写的时候，要加上newline=''和dialect='excel'，否则每行后面会增加一个空白行
csv_writer=csv.writer(open('./data/results_daily.csv','w',encoding='utf-8',newline=''),dialect='excel')
title = ['time','uid','pid','detail']
csv_writer.writerow(title)

insertStr = ""
detailList = []
for i, sub in enumerate(submissions):
    date = sub[0]
    user = sub[1]
    problem = sub[2]
    clock = sub[3]
    result = sub[4]
    code = str(sub[5])
    detail = (clock, code_transform_list[code], resultDic[result])
    if i<len(submissions)-1:
        nextDate = submissions[i + 1][0]
        nextUser = submissions[i+1][1]
        nextProblem = submissions[i+1][2]
    else:
        nextUser = 0
        nextProblem = 0
        nextDate = 0
    detailList.append(detail)
    if user == nextUser and problem == nextProblem and date == nextDate:
        continue
    else:
        #insert = """INSERT: time-'{}', user-{}, problem-{}, detail-{}""".format(date, user, problem, detailList)
        row = [date.strftime('%Y-%m-%d'), user_transform_list[user], problem_transform_list[problem], detailList]
        print(row)
        csv_writer.writerow(row)
        detailList = []



import sqlConnect as sc
import IdTransformer as it
import csv
import pandas as pd
import json

conn = sc.sql_connect()
transformer = it.IdTransformer()

'''重新编码题目，加入time_limit和memory_limit，并更新在数据库'''
def recode_problem():
    problem_ids = conn.select("select pid from n_problems")
    new_problem_ids = transformer.transform(problem_ids, 'problem')
    problem_limits = conn.select("SELECT id, test_setting FROM problems")
    p_limit = {}
    for p in problem_limits:
        p_limit[p[0]] = p[1]
    print(p_limit)

    for i, pid in enumerate(problem_ids):
        old_id = pid[0]
        new_id = new_problem_ids[i]
        limit = p_limit[old_id]
        limit_json = json.loads(limit)
        time_limit = limit_json['time_limit']
        memory_limit = limit_json['memory_limit']
        update = """update n_problems set pid_='{}',timelimit={},memorylimit={} where pid = {}""".format(new_id, time_limit, memory_limit, old_id)
        print(update)
        conn.update(update)


'''重新编码比赛并更新数据库'''
def recode_contest():
    contest_ids = conn.select("select cid from n_contests ORDER BY stime")
    new_contest_ids = transformer.transform(contest_ids, 'contest')
    contest_transform_list = {}
    for i, cid in enumerate(contest_ids):
        contest_transform_list[cid[0]] = new_contest_ids[i]
        update = """update n_contests set cid_='{}' where cid = {}""".format(new_contest_ids[i], cid[0])
        conn.update(update)
    print(contest_transform_list)


'''重新编码知识点并更新数据库'''
def recode_knowledg():
    knowledge_ids = conn.select("select kid from d_tags")
    new_knowledge_ids = transformer.transform(knowledge_ids, 'knowledge')
    knowledge_transform_list = {}
    for i, kid in enumerate(knowledge_ids):
        knowledge_transform_list[kid[0]] = new_knowledge_ids[i]
        update = """update d_tags set kid_='{}' where kid = {}""".format(new_knowledge_ids[i], kid[0])
        print(update)
        conn.update(update)
    print(knowledge_transform_list)


'''重新编码用户'''
def recode_user():
    users = conn.select("select uid, college from n_users")
    new_user_ids = transformer.transform(users, 'user')
    colleges = conn.select("select distinct college from users")
    new_college_ids = transformer.transform(colleges, 'college')
    college_transform_list = {}
    for i, cid in enumerate(colleges):
        college_transform_list[cid[0]] = new_college_ids[i]
    for i, user in enumerate(users):
        uid = user[0]
        college = college_transform_list[user[1]]
        new_uid = new_user_ids[i]
        update = """update n_users set uid_='{}', college_='{}' where uid = {}""".format(new_uid, college, uid)
        print(update)
        conn.update(update)


'''重新编码code记录表并提取codes.csv，由于比较多，就先编码后存在csv文件里，然后读取为字典'''
def recode_save_code():
    # 重新编码code
    print('selecting...')
    code_ids = conn.select("select id from submission_codes")
    print('transforming...')
    new_code_ids = transformer.transform(code_ids, 'code')

    # 太多了，记录一下重新编码的code
    csv_writer=csv.writer(open('./data/code_transform.csv','w',encoding='utf-8',newline=''),dialect='excel')
    for i,codeid in enumerate(code_ids):
        row = [codeid[0], new_code_ids[i]]
        print(row)
        csv_writer.writerow(row)


'''从csv读取编码结果，查询代码数据并保存'''
def save_codes():
    # 读取重新编码的code结果，记录在字典里，便于查询
    csv_reader = csv.reader(open('./data/code_transform.csv','r'))
    code_transform_list = {}
    print('reading...')
    for row in csv_reader:
        code_transform_list[row[0]] = row[1]
    print(code_transform_list)

    # 读取code记录并保存
    print('reading submissions and codes...')
    code_list = conn.select("""SELECT sc.id as codeid, s.lang as language, s.time_cost as runtime, s.memory_cost as memory, sc.content as code
                            FROM submission_codes sc, submissions_all s
                            WHERE sc.submission_id=s.id
                            AND s.creator_id in (SELECT uid FROM n_users)
                            AND s.problem_id in (SELECT pid FROM n_problems)
                            AND (s.contest_id is NULL OR s.contest_id in (SELECT cid FROM n_contests))
                            """)

    csv_writer=csv.writer(open('./data/solutions.csv','w',encoding='utf-8',newline=''),dialect='excel')
    title = ['solutionid','code','language','runtime','memory']
    csv_writer.writerow(title)
    for code in code_list:
        codeid = code[0]
        codeid_ = code_transform_list[str(codeid)]
        language = code[1]
        runtime = code[2]
        memory = code[3]
        code = code[4]
        row = [codeid_, code, language, runtime, memory]
        print(row)
        csv_writer.writerow(row)


'''读codes.csv文件（太大了直接打不开），有空行有点烦，需要replace一下'''
def read_codes():
    # 注意python3这里要用'r'，一些教程介绍让用'rb'，会出错
    with open('./data/codes_710.csv', 'r', encoding='gb18030') as f:
        data = csv.reader((line.replace('\0', '') for line in f), delimiter=",")
        i = 0
        for row in data:
            i += 1
            print(i)

'''改code.csv编码，并把id改成SXXX这样'''
def change_codes():
    csv_writer = csv.writer(open('./data/solutions.csv','w',encoding='utf-8',newline=''),dialect='excel')
    with open('./data/codes.csv', 'r', encoding='gb18030') as f:
        data = csv.reader((line.replace('\0', '') for line in f), delimiter=",")
        for row in data:
            row[0] = row[0].replace('C','S')
            print(row)
            csv_writer.writerow(row)

if __name__ == '__main__':
    #recode_user()
    #save_codes()
    #recode_contest()
    #change_codes()
    #recode_contest()
    save_codes()
    print('ok')
import sqlConnect
import csv
import TransformLists as tl
import pandas as pd

transformer = tl.TransformLists()
#user_transform_list = transformer.get_transform_list('user')
conn = sqlConnect.sql_connect()

def save_users():
    csv_writer=csv.writer(open('./data/users.csv','w',encoding='utf-8',newline=''),dialect='excel')
    title = ['uid','college','grade']
    csv_writer.writerow(title)

    #users = conn.select("""SELECT user_id, student_id FROM user_authenticate WHERE student_id NOT LIKE '__00%' and student_id LIKE '1%' ORDER BY user_id""")
    users = conn.select ("""SELECT uid_, college_, student_id FROM n_users""")

    for user in users:
        uid = user[0]
        college = user[1]
        grade = '20'+user[2][:2]
        row = [uid, college, grade]
        print(row)
        csv_writer.writerow(row)


# 稍微占位置写一下result.csv这个文件，直接用excel手写会有编码问题
def write_results():
    csv_writer = csv.writer(open('./data/results.csv','w',encoding='utf-8',newline=''),dialect='excel')
    rows = [
        ['rid','rname'],
        [1,'Accepted'],
        [2,'Compile Error'],
        [3,'IFNR'],
        [4,'JG'],
        [5,'Memory Limit Exceed'],
        [6,'Presentation Error'],
        [7,'Runtime Error ( erroneous arithmetic operation ）'],
        [8,'Runtime Error ( make an invalid virtual memory reference or segmentation fault)'],
        [9,'Time Limit Exceeded'],
        [10,'Wrong Answer'],
        [11,'WT'],
        [12,'Other Error']
    ]
    for row in rows:
        csv_writer.writerow(row)

if __name__ == '__main__':
    #write_results()
    #save_users()
    data = pd.read_csv('./data/solutions.csv')
    print(data)
import pymysql
class sql_connect():
    def __init__(self):
        self.get_connect()

    def get_connect(self):
        try:
            self.conn=pymysql.connect(host='localhost', user='root', password='123456', database='oj4th')
            self.cursor=self.conn.cursor()
            print("已连接")
        except Exception as e:
            print(e)

    def select(self,select_str,ind=0):
        data=[]
        try:
            self.cursor.execute(select_str)
            data=self.cursor.fetchall()
        except Exception as e:
            print(e,ind)
        if data==[]:
            return [('null',)]
        return data

    def insert(self,insert_str):
        try:
            self.cursor.execute(insert_str)
            self.conn.commit()
        except Exception as e:
            print(e)

    def update(self,update_str):
        try:
            self.cursor.execute(update_str)
            self.conn.commit()
        except Exception as e:
            print(e)

    def create(self,create_str):
        try:
            self.cursor.execute(create_str)
            self.conn.commit()
        except Exception as e:
            print(e)

    def close_conn(self):
        try:
            if self.conn is not None:
                self.cursor.close()
                self.conn.close()
                print("数据库已断开")
        except Exception as e:
            print(e)


    #获取一个列表，保证查询语句返回的是一列内容，即要获取的列表
    def select_list(self,select_4_list):
        list = ''
        try:
            data_list=self.select(select_4_list)
            for l in data_list:
                list=list+str(l[0])+','
            list=list[:-1]
        except Exception as e:
            print(e)
        return list



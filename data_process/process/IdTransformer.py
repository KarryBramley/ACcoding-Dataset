#coding: utf-8
import numpy as np
from numpy import random
import pandas as pd

class IdTransformer:
    def __init__(self):
        return

    # Transform id list into new encoded id list
    # Input:
    # id_list: numpy array
    # type: str
    #
    # Output:
    # ret_id_list: list

    def transform(self, id_list, type):
        #id_num = id_list.shape[0]
        id_num = len(id_list)
        arr = np.arange(0, id_num)
        base, offset_max = 0, 0
        signature = ''
        if type == 'knowledge':
            arr = random.permutation(arr)
            base = random.randint(500000, 550000)
            offset_max = 10
            signature = 'K'
        elif type == 'problem':
            arr = random.permutation(arr)
            base = random.randint(200000, 300000)
            offset_max = 20
            signature = 'P'
        elif type == 'contest':
            # arr = random.permutation(arr)
            base = random.randint(600000, 700000)
            offset_max = 15
            signature = 'C'
        elif type == 'user':
            arr = random.permutation(arr)
            base = random.randint(100000, 200000)
            offset_max = 12
            signature = 'U'
        elif type == 'college':
            arr = random.permutation(arr)
            base = random.randint(800000, 900000)
            offset_max = 15
            signature = 'CL'
        elif type == 'code':
            # Because the original code id list is organized in time order, permutation would disorder the id list
            base = random.randint(3100000, 3600000)
            offset_max = 5
            signature = 'S'
        # print(base)
        ret_arr = np.zeros(id_num, dtype=np.int)
        for i, num in enumerate(arr):
            offset = random.randint(1, offset_max)
            ret_arr[num] = base + offset
            base = base + offset
        ret_id_list = [signature + str(id) for id in ret_arr]
        return ret_id_list


if __name__ == '__main__':
    transformer = IdTransformer()
    id_list = np.arange(1, 101)
    # print(id_list)
    new_id_list = transformer.transform(id_list, type='code')
    # print(id_list)
    print(new_id_list)
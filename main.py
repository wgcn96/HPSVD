# -*- coding: utf-8 -*-

"""
note something here

"""


import math
import time


import numpy as np
np.random.seed(0)
import pandas as pd

from hawkes import TSHawkes


__author__ = 'Wang Chen'
__time__ = '2019/7/9'


if __name__ == '__main__':
    start = time.time()
    print(start)
    users_num = 1000
    data_path = 'D:\\workData\\tecent_for_hawkes\\data{}\\'.format(users_num)
    level = 5
    UIT = pd.read_csv(data_path + 'train.csv', header=None)    # 0:user 1:item 2:time 3:video_type 4:country 5:province 6:city
    print(UIT.head())

    cluster_num = np.array([1])
    u_cluster_m = np.zeros((users_num, 1), dtype=np.int)

    test_datasets = pd.read_csv(data_path + 'test.csv', header=None)
    test_datasets[0] = test_datasets[level]
    test_datasets = test_datasets.drop([4, 5, 6], axis=1)
    item_c = np.zeros(max(max(UIT[1]), max(test_datasets[1])) + 1, dtype=np.int)

    UIT[0] = UIT[level]
    UIT = UIT.drop([4, 5, 6], axis=1)
    print(UIT.head())


    penalty_bu = 0
    penalty_bi = 1e8
    penalty_p = 1e3
    penalty_q = 1e3
    k1 = 0.5
    k2 = 0.5
    penalty_beta = 0

    re = []
    para = []
    obj = []

    # for penalty_bi in [1e9,1e2]:
    #     for penalty_p in [1e5 , 1e3]:
    #         for penalty_q in [1e2 , 1e1,2]:
    #             for k2 in [0.3,0.6]:
    #                 for k1 in [0.3,0.6]:
    print((k1, k2, penalty_bu, penalty_bi, penalty_p, penalty_q, penalty_beta))
    ts = TSHawkes(UIT.values, 24, k1, k2, 3, penalty_bu, penalty_bi, penalty_p, penalty_q, penalty_beta, u_cluster_m,
                  cluster_num + 1, item_c, test_datasets, if_print=True, validate=True)
    ts.fit_with_BFGS()
    re.append(ts.validate_test())
    para.append((k1, k2, penalty_bu, penalty_bi, penalty_p, penalty_q, penalty_beta))
    ts.likelihood_()
    obj.append(ts.obj_pre[len(ts.obj_pre) - 1])
    print("finish, total time: {}".format(time.time()-start))

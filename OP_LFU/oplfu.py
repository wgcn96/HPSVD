# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/7/19'

import time
from queue import PriorityQueue

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy import integrate

from OP_LFU.Item import Item
from OP_LFU.static import *

class OPLFU:
    """
    UIT_dataframe
    item_num
    test_day
    item_list
    func_num
    cache_size
    cache_list
    cache_set
    """
    def __init__(self, dataframe, item_num, test_day, func_num, cache_size, if_disp = False):
        self.dataframe = dataframe
        self.item_num = item_num
        self.test_day = test_day
        self.func_num = func_num
        self.cache_size = cache_size
        self.item_list = self._Item_list(item_num, func_num)
        for item_id, item_group in self.dataframe.groupby([1]):    ### leave a hard code for column index
            watch_time_df = item_group[2]
            watch_time_vec, item_cdf_vec = self._item_cdf(watch_time_df)
            self.item_list[item_id].watch_time_vec = watch_time_vec
            self.item_list[item_id].item_cdf_vec = item_cdf_vec

        self.cache_list = PriorityQueue()
        self.cache_set = set()

        self.if_disp = if_disp

    def _Item_list(self, item_num, func_num):
        list = [Item(i, func_num) for i in range(item_num)]
        return list

    def _item_cdf(self, watch_time_df):
        total_count = len(watch_time_df)
        time_vec = np.arange(start=0, stop=30, dtype=np.int32)
        cdf_vec = np.zeros((30,), dtype=np.int32)
        pos_list = []
        day_count_list = []
        for pos, day_count in watch_time_df.groupby(watch_time_df):
            pos_list.append(pos)
            day_count_list.append(len(day_count))
        first_order = 0
        first_num = 0
        for i in range(len(day_count_list)):
            cdf_vec[first_order:pos_list[i]] = first_num
            first_order = pos_list[i]
            first_num += day_count_list[i]
        cdf_vec[first_order:self.test_day] = first_num
        assert total_count == first_num     # debug use
        return time_vec, cdf_vec
    
    def estimate_test_day(self, time, para, func_type):
        def func_cons():
            f = (para[0] * time - para[1])
            return f

        def func_power():
            f = para[0] * np.power(time, para[1])
            return f

        def func_exp():
            f = para[0] * (1 - np.power(np.e, -1 * para[1] * time))
            return f

        func_list = [func_cons, func_power, func_exp]
        return func_list[func_type]()

    def train_plus_test(self):
        if self.if_disp:
            start = time.time()
            print("current test day {}, begin to train the model".format(self.test_day))

        for item_id, item in enumerate(self.item_list):

            if self.if_disp:
                print("current item id: {}".format(item_id))

            if item.watch_time_vec is None:
                item.estimate = -np.inf
                continue

            watch_time_vec = item.watch_time_vec[:self.test_day]
            item_cdf_vec = item.item_cdf_vec[:self.test_day]

            def func_cons(x):
                """
                :param x: para_vec [a,b]
                :return:
                """
                f = item_cdf_vec - (x[0] * watch_time_vec - x[1])
                return f

            def func_power(x):
                f = item_cdf_vec - (x[0] * np.power(watch_time_vec, x[1]))
                return f

            def func_exp(x):
                f = item_cdf_vec - (x[0] * (1 - np.power(np.e, -1 * x[1] * watch_time_vec)))
                return f

            func_list = [func_cons, func_power, func_exp]
            result_err = np.zeros((self.func_num,), np.float32)
            param_list = []
            for i in range(self.func_num):
                # solver = root(func_list[i], x0=np.array([1, 1]), method='lm')
                solver = root(func_list[i], x0=np.random.randn(2), method='lm')
                result_err[i] = np.linalg.norm(solver.fun)
                param_list.append(solver.x)
            chose = np.argsort(result_err)[0]
            param = param_list[chose]

            item.type = chose
            item.param = param
            
            # item.estimate = func_list[item.type]()
            item.estimate = self.estimate_test_day(self.test_day, item.param, item.type) - item.item_cdf_vec[self.test_day-1]

            if self.if_disp:
                print("item {} estimate {}".format(item_id, item.estimate))

        if self.if_disp:
            print("train and test finish, total time {}".format(time.time() - start))

    def cache_and_validate(self):
        for item in self.item_list:
            if len(self.cache_set) < self.cache_size:
                self.cache_set.add(item.id)
                self.cache_list.put((item.estimate, item.id))
            else:
                (top_priority, top_item) = self.cache_list.queue[0]

                if item.estimate >= top_priority:  # 替换条件
                    (replace_popularity, replace_item) = self.cache_list.get()
                    self.cache_set.remove(replace_item)
                    self.cache_list.put((item.estimate, item.id))
                    self.cache_set.add(item.id)

        hit = 0
        count = 0
        current_testday_items = self.dataframe[self.dataframe[2] == self.test_day][1].values        ### leave a hard code for column index
        for test_item in current_testday_items:
            count += 1
            if test_item in self.cache_set:
                hit += 1
        return (hit, count)


if __name__ == '__main__':
    part_one = pd.read_csv(data_path + 'train.csv', header=None)
    part_two = pd.read_csv(data_path + 'test.csv', header=None).drop([7], axis=1)
    UIT = pd.concat([part_one, part_two], axis=0)

    level = 0
    result_list = []
    group_count = 0

    for cache_size in [10, 25, 50, 75, 100, 250, 500]:
        print("current cache_size {}".format(cache_size))
        total_hit = 0
        total_count = 0
        for name, group in (UIT.groupby([level])):
            group_count += 1
            items_num = max(group[1] + 1)
            for test_day in range(24, 30, 1):
                # print("current test day {}".format(test_day))
                oplfu = OPLFU(group, items_num, test_day, 3, cache_size=cache_size, if_disp=False)
                oplfu.train_plus_test()

                hit, count = oplfu.cache_and_validate()
                total_hit += hit
                total_count += count
                # print("hit and count: {}, {}".format(hit, count))
        result_list.append(round(total_hit/total_count, 4))
        print("current hit rate {}".format(round(total_hit/total_count, 4)))
    print(result_list)

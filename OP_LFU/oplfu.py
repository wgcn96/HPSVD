# -*- coding: utf-8 -*-

"""
main file for OPLFU paper

--文件调用关系：
    Item：视频类
    script：相当于util，主要是实现积分时在该文件测试
    static：静态路径文件，替换为自己的代码即可运行
    __main__：主函数入口
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
    王臣实现的OPLFU版本
    核心思想：当前测试天由前面所有天训练数据训练模型，选择最优模型，进行popularity预测和缓存
             缓存结束后验证 hit rate

    @class members

    UIT_dataframe : pandas dataframe (用户，视频，时间）数据
    item_num : 视频数量
    test_day : 当前测试天（例如前24天训练，第二十五天测试，则下标从0开始 test_day=24 ）
    item_list : 所有视频的列表，注意视频用Item类的对象表示
    func_num : 待选函数数量，本实现中func_num为4，分别为 常数 指数 幂函数 高斯函数
    cache_size : 缓存列表大小
    cache_list : 缓存列表
    cache_set : 缓存集合（跟列表功能一样，为了快速查找）
    if_disp : 输出，debug


    @class funcs

    train_plus_test : 训练当前test_day之前所有item的popularity拟合函数并记录

    estimate_test_day : 评估当前test_day所有item此时的popularity

    cache_and_validate : 缓存并验证是否击中

    update_one_day : test_day += 1 （前进一天）
    """

    def __init__(self, dataframe, item_num, test_day, func_num, cache_size, max_day=30, if_disp=False):
        self.dataframe = dataframe
        self.item_num = item_num
        self.test_day = test_day
        self.max_day = max_day
        self.func_num = func_num
        self.cache_size = cache_size
        self.item_list = self._Item_list(item_num, func_num)
        for item_id, item_group in self.dataframe.groupby([1]):  ### leave a hard code for column index
            watch_time_df = item_group[2]
            watch_time_vec, item_cdf_vec = self._item_cdf(watch_time_df)
            self.item_list[item_id].watch_time_vec = watch_time_vec
            self.item_list[item_id].item_cdf_vec = item_cdf_vec

        self.cache_list = PriorityQueue()
        self.cache_set = set()

        self.if_disp = if_disp

    def _Item_list(self, item_num, func_num):
        """
        初始化所有视频（Item）为类实例
        :param item_num:
        :param func_num:
        :return:
        """
        list = [Item(i, func_num) for i in range(item_num)]
        return list

    def _item_cdf(self, watch_time_df):
        """
        生成两个列表，时间（天）列表和对应的观看次数列表
        :param watch_time_df:
        :return:
        """
        total_count = len(watch_time_df)
        time_vec = np.arange(start=0, stop=self.max_day, dtype=np.int32)
        cdf_vec = np.zeros((self.max_day,), dtype=np.int32)
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
        cdf_vec[first_order:] = first_num
        assert total_count == first_num  # debug use
        return time_vec, cdf_vec

    @staticmethod
    def inte_gauss(x, mu, sigma):
        """
        高斯积分
        静态函数，可以放在类外面
        :param x:
        :param mu:
        :param sigma:
        :return:
        """
        return np.power(np.e, -1 * (x - mu) * (x - mu) / (2 * sigma * sigma)) / np.sqrt(2 * np.pi * sigma * sigma)

    def estimate_test_day(self, time, para, func_type):
        """
        评估当前test_day所有item此时的popularity
        :param time:
        :param para:
        :param func_type:
        :return:
        """
        def func_cons():
            f = (para[0] * time - para[1])
            return f

        def func_power():
            f = para[0] * np.power(time, para[1])
            return f

        def func_exp():
            f = para[0] * (1 - np.power(np.e, -1 * para[1] * time))
            return f

        def func_gauss():
            term_sigma = 2 * para[1] * para[1]
            regularization = 1 / np.sqrt(np.pi * term_sigma)
            integrate_result, err = integrate.quad(self.inte_gauss, -np.inf, time, args=(para[0], para[1]))
            f = regularization * integrate_result
            return f

        func_list = [func_cons, func_power, func_exp, func_gauss]
        return func_list[func_type]()

    def train_plus_test(self):
        """
        训练当前test_day之前所有item的popularity拟合函数并记录
        :return:
        """
        if self.if_disp:
            start = time.time()
            print("current test day {}, begin to train the model".format(self.test_day))

        previous_items = self.dataframe[self.dataframe[2] < self.test_day][1].unique()

        for item_id in previous_items:
            item = self.item_list[item_id]

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

            def func_gaussian(x):
                f = []
                term_sigma = 2 * x[1] * x[1]
                regularization = 1 / np.sqrt(np.pi * term_sigma)
                for item, time in zip(item_cdf_vec, watch_time_vec):
                    integrate_result, err = integrate.quad(self.inte_gauss, -np.inf, time, args=(x[0], x[1]))
                    f_item = item - regularization * integrate_result
                    f.append(f_item)
                f = np.array(f)
                return f

            func_list = [func_cons, func_power, func_exp, func_gaussian]
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
            item.estimate = self.estimate_test_day(self.test_day, item.param, item.type) - item.item_cdf_vec[
                self.test_day - 1]

            if self.if_disp:
                print("item {} estimate {}".format(item_id, item.estimate))

        if self.if_disp:
            print("train and test finish, total time {}".format(time.time() - start))

    def cache_and_validate(self):
        """
        先进先出缓存队列并验证是否击中
        :return:
        """
        previous_items = self.dataframe[self.dataframe[2] < self.test_day][1].unique()

        for item_id in previous_items:
            item = self.item_list[item_id]
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
        current_testday_items = self.dataframe[self.dataframe[2] == self.test_day][
            1].values  ### leave a hard code for column index
        for test_item in current_testday_items:
            count += 1
            if test_item in self.cache_set:
                hit += 1
        return (hit, count)

    def update_one_day(self):
        """
        天数加1
        :return:
        """
        self.test_day += 1
        self.cache_list = PriorityQueue()
        self.cache_set = set()
        if self.test_day > self.max_day:
            raise Exception("days exceed")

        if self.if_disp:
            print("one day adances, current day {}".format(self.test_day))

        # return self.cur_day


if __name__ == '__main__':
    part_one = pd.read_csv(data_path + 'train.csv', header=None)
    # part_two = pd.read_csv(data_path + 'test.csv', header=None).drop([7], axis=1)
    part_two = pd.read_csv(data_path + 'test.csv', header=None)
    UIT = pd.concat([part_one, part_two], axis=0)       # 生成训练测试数据

    level = 0
    print(level)
    print(data_path)
    result_list = []
    group_count = 0

    # 不同缓存大小
    for cache_size in [10, 25, 50, 75, 100, 250, 500]:
        print("current cache_size {}".format(cache_size))
        total_hit = 0
        total_count = 0
        for name, group in (UIT.groupby([level])):      # 缓存层次（国家， 地区， 等）
            print("current user {}".format(name))
            group_count += 1
            items_num = max(group[1] + 1)
            oplfu = OPLFU(group, items_num, test_day=24, func_num=3, cache_size=cache_size, if_disp=True)   # 开始训练
            for test_day in range(24, 30, 1):
                oplfu.train_plus_test()     # 训练
                hit, count = oplfu.cache_and_validate()     # 验证
                total_hit += hit        # 结果
                total_count += count
                oplfu.update_one_day()  # 天数加1
                # print("hit and count: {}, {}".format(hit, count))
        result_list.append(round(total_hit / total_count, 4))
        print("current hit rate {}".format(round(total_hit / total_count, 4)))
    print(result_list)

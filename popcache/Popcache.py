# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/7/15'

import copy
from queue import PriorityQueue

import pandas as pd
import numpy as np

from popcache.Item import Item
from popcache.Event import Event
from popcache.Hypercube import Hypercube
from popcache.static import *

# demo for one day

class Popcache:
    """
    dim: 维度
    item_list: 视频列表，主键为视频的id
    hypercube_array: 立方空间的dim维数组
    max_day: 最大天数
    cur_day: 当天天数
    all_UITs: np.adarray 从pandas转化来的原始值
    event_lastday_list: 前一天的事件缓存列表
    event_count: 事件计数
    cache_size: 缓存列表大小
    cache_list: 缓存列表，存储视频id
    cache_set:

    if_disp: 是否展示
    """
    def __init__(self, dim, max_item_id, max_day, cur_day, all_UITs, cache_size, if_disp):
        """
        dim: 维度
        max_item_id: 最大视频id
        max_day: 最大天数
        cur_day: 当天天数
        all_UITs: np.adarray 从pandas转化来的原始值
        event_lastday_list: 前一天的事件缓存列表
        cache_size: 缓存列表大小

        if_disp: 是否展示
        """
        self.dim = dim
        self.item_list = [Item(item_id) for item_id in range(max_item_id)]
        self.hypercube_array = self.init_hypercube_array(dim)
        self.max_day = max_day
        self.cur_day = cur_day
        self.all_UITs = all_UITs
        self.curday_values = None
        self.event_lastday_list = []
        self.event_count = -1
        self.cache_size = cache_size
        self.cache_list = PriorityQueue()
        self.cache_set = set()

        self.if_disp = if_disp

    def init_hypercube_array(self, dim):
        """
        构建hypercube空间
        :param dim:
        :return:
        """
        hypercube_array = [Hypercube() for _ in range(2)]
        for _ in range(dim - 1):
            hypercube_array = [copy.deepcopy(hypercube_array) for _ in range(2)]
        return hypercube_array

    def enumerate_hypercubes(self):
        hypercubes_list = self.hypercube_array
        for depth in range(self.dim-1):
            tmp_list = []
            for element in hypercubes_list:
                tmp_list.extend(element)
            hypercubes_list = tmp_list

        if self.if_disp:
            print("enumerate {} depth, total length {}".format(dim, hypercubes_list.__len__()))

        return hypercubes_list

    def get_one_day_values(self):
        """
        获取一天的ndarray
        :return:
        """
        filter_UITs = self.all_UITs[self.all_UITs[:, 3] == self.cur_day]

        if self.if_disp:
            print("cur day {}".format(self.cur_day), end=" , ")
            print("total events {}".format(filter_UITs.shape[0]))
        return filter_UITs

    def extract_feature(self, item_id):
        """
        通过item_id获取视频的特征
        :param item_id:
        :return:
        """
        if self.cur_day > self.max_day:
            raise Exception("days exceed")

        cur_item = self.item_list[item_id]
        if np.sum(cur_item.history) == 0:
            return np.zeros((self.dim,), dtype=np.int32)

        feature = [0] * self.dim
        for pos, feature_day in enumerate(feature_day_list):
            if self.cur_day >= feature_day:
                check = np.sum(cur_item.history[self.cur_day - feature_day:self.cur_day])
            else:
                check = np.sum(cur_item.history[0:self.cur_day])
            if check > 0:
                feature[pos] = 1

        if self.if_disp:
            print("get the feature: ", np.array(feature, dtype=np.int32))

        return np.array(feature, dtype=np.int32)

    def update_feature(self):
        """
        根据ndarray更新当天视频的特征值
        :return:
        """
        count = 0
        for value_line in self.curday_values:
            item_id = value_line[2]
            self.item_list[item_id].history[self.cur_day] += 1
            count += 1

        item_count = 0
        for item in self.item_list:
            if item.history[self.cur_day]:
                item.popularity = item.history[self.cur_day]
                item_count += 1

        if self.if_disp:
            print("total deal with {} events, update {} items".format(count, item_count))

        return item_count

    def update_hypercube_estimate_value(self):
        """
        根据event_list列表更新hypercube他们的MN值
        :return:
        """

        total_popularity = 0

        if self.if_disp:
            print("update hypercube M N")

        for event in self.event_lastday_list:
            video_id = event.item
            event_feature = self.extract_feature(video_id)
            cube = self.select_hypercube(event_feature)
            popularity_plus = cube.update_popularity(self.item_list[video_id].popularity)
            total_popularity += popularity_plus

            if self.if_disp:
                print("current hypercube ", event_feature, end=" , ")
                print("popularity plus {}".format(popularity_plus))

        return total_popularity

    def update_cache_set(self):
        tmp_list = PriorityQueue()
        while not self.cache_list.empty():
            (old_priority, old_item) = self.cache_list.get()
            event_feature = self.extract_feature(old_item)
            cube = self.select_hypercube(event_feature)
            update_estimate_popularity = cube.get_popularity()
            tmp_list.put((update_estimate_popularity, old_item))

        if tmp_list.qsize() != self.cache_set.__len__():
            raise Exception("update heap error")

        self.cache_list = tmp_list
        return True

    def update_one_day(self):
        """
        天数加1
        :return:
        """
        self.cur_day += 1
        self.event_lastday_list = []
        if self.cur_day > self.max_day:
            raise Exception("days exceed")

        if self.if_disp:
            print("one day adances, current day {}".format(self.cur_day))

        return self.cur_day

    def select_hypercube(self, feature):
        """
        根据视频的特征选择对应的hypercube
        :param feature:
        :return:
        """
        hypercube = self.hypercube_array
        for i in feature:
            hypercube = hypercube[i]

        assert isinstance(hypercube, Hypercube)

        if self.if_disp:
            print("current hypercube: ", feature)

        return hypercube

    def estimate_popularity(self, hypercube, event):
        """
        根据hypercube的总体估计当前事件的流行度
        :param hypercube:
        :param event_id:
        :return:
        """
        hypercube.add_event(event)       # 添加一个事件
        popularity = hypercube.get_popularity()     # 获取popularity
        event.esti_popularity = popularity

        if self.if_disp:
            print("the esti_popularity is {}".format(popularity))

        return popularity

    def event_add_oneday(self):
        """
        添加cur_day当天的event进event_list
        """
        self.curday_values = self.get_one_day_values()
        for value_line in self.curday_values:
            self.event_count += 1
            event = Event(id=value_line[0], user=value_line[1], item=value_line[2], occur_time=value_line[3], esti_popularity=0)
            self.event_lastday_list.append(event)

        if self.if_disp:
            print("today add {} events, current event pos: {}".format(self.curday_values.shape[0], self.event_count))    # for check use

        return self.event_lastday_list

    def curday_event_into_cube(self, is_validate=False):
        count = 0
        hit = 0
        for event in self.event_lastday_list:

            if self.if_disp:
                print("estimate event {}".format(event.id))

            event_feature = self.extract_feature(event.item)
            cube = self.select_hypercube(event_feature)
            _estimate_popularity = self.estimate_popularity(cube, event)

            hit += self.metric(event)
            count += 1

        if self.if_disp:
            print("total estimate {} events".format(count))

        if is_validate:
            print("today hit rate is {}".format(round(hit/count, 3)))

        return count, hit

    def print_cubes(self):
        print("day {} show cubes".format(self.cur_day))
        hypercubes_list = self.enumerate_hypercubes()

        for pos, hypercube in enumerate(hypercubes_list):
            print("current cube {}".format(pos), end=" , ")
            print("M:{} N:{}".format(hypercube.M, hypercube.N), end=" , ")
            print(hypercube.get_popularity())

    def metric(self, event):
        """
        最小堆比较，缓存击中返回1，否则返回0；
        当缓存不中时，同时需要调整最小堆。
        :param event:
        :return:
        """
        hit = 0
        if event.item not in self.cache_set:
            if len(self.cache_set) < self.cache_size:
                self.cache_set.add(event.item)
                self.cache_list.put((event.esti_popularity, event.item))
            else:
                (top_priority, top_item) = self.cache_list.queue[0]

                if event.esti_popularity >= top_priority:  # 替换条件
                    (replace_popularity, replace_item) = self.cache_list.get()
                    self.cache_set.remove(replace_item)
                    self.cache_list.put((event.esti_popularity, event.item))
                    self.cache_set.add(event.item)
                    # print("{} out and {} in".format(replace_popularity, event.esti_popularity))
            return hit
        else:
            hit = 1
            return 1



if __name__ == '__main__':
    part_one = pd.read_csv(data_path + 'train.csv', header=None)
    part_two = pd.read_csv(data_path + 'test.csv', header=None).drop([7], axis=1)
    all_UITs = pd.concat([part_one, part_two], axis=0).values
    items_num = max(all_UITs[:, 1] + 1)
    index = np.arange(0, all_UITs.shape[0]).reshape((all_UITs.shape[0], 1))
    all_UITs = np.column_stack((index, all_UITs))

    for cache_size in [5, 10] + [int(items_num * ratio) for ratio in [0.001, 0.0025, 0.005, 0.01]]:
        popcache = Popcache(dim, items_num, days, cur_day=0, all_UITs=all_UITs, cache_size=cache_size, if_disp=False)
        for i in range(24):
            event_list = popcache.event_add_oneday()
            count, hit = popcache.curday_event_into_cube()
            item_count = popcache.update_feature()
            total_popularity = popcache.update_hypercube_estimate_value()
            popcache.update_cache_set()
            # popcache.print_cubes()
            # popcache.enumerate_hypercubes()
            popcache.update_one_day()

        validate_list = []
        for i in range(24, 30, 1):
            event_list = popcache.event_add_oneday()
            count, hit = popcache.curday_event_into_cube(is_validate=True)
            validate_list.append(hit/count)
            item_count = popcache.update_feature()
            total_popularity = popcache.update_hypercube_estimate_value()
            popcache.update_cache_set()
            # popcache.print_cubes()
            # popcache.enumerate_hypercubes()
            popcache.update_one_day()
        print(validate_list, np.array(validate_list).mean())

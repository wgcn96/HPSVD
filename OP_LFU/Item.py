# -*- coding: utf-8 -*-

"""
class for Item

视频类

"""

__author__ = 'Wang Chen'
__time__ = '2019/7/20'


class Item:
    """
    id: 视频编号
    type: 拟合函数种类 （0 1 2 3）
    param: 拟合函数参数
    watch_time_vec: 时间列表
    item_cdf_vec: 时间列表对应的观看频次cdf列表
    """

    def __init__(self, id, type):
        self.id = id
        self.type = type
        self.param = None
        self.watch_time_vec = None
        self.item_cdf_vec = None
        self.estimate = 0





if __name__ == '__main__':
    pass
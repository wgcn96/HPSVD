# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/7/19'

import sys
import math

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy import integrate

from OP_LFU.static import *


def item_cdf(watch_time_vec):
    cdf_vec = np.arange(1, len(watch_time_vec) + 1)
    return cdf_vec

def func(x):
    return x + 2 * np.cos(x)


def func2(x):
    f = [x[0] * np.cos(x[1]) - 4,
         x[1]*x[0] - x[1] - 5]
    # df = np.array([[np.cos(x[1]), -x[0] * np.sin(x[1])], [x[1], x[0] - 1]])
    return f


def func_cons(x):
    """

    :param x: para_vec [a,b]
    :return:
    """
    f = item_cdf_vec - (x[0]*watch_time_vec-x[1])
    return f

def func_power(x):
    f = item_cdf_vec - (x[0] * np.power(watch_time_vec, x[1]))
    return f


def func_exp(x):
    f = item_cdf_vec - (x[0] * (1 - np.power(np.e, -1*x[1]*watch_time_vec)))
    return f


def func_gauss(x):
    term_sigma = 2 * x[1] * x[1]
    regularization = 1/np.sqrt(np.pi*term_sigma)
    # term


def inte_gauss(x, mu, sigma):
    return np.power(np.e, -1*(x-mu)*(x-mu)/(2*sigma*sigma))/np.sqrt(2*np.pi*sigma*sigma)



if __name__ == '__main__':
    part_one = pd.read_csv(data_path + 'train.csv', header=None)
    item_group = []
    for name, group in part_one.groupby([1]):
        item_group.append(group)
    name = 2
    item_chose = item_group[2]
    watch_time_vec = item_chose[2].values
    item_cdf_vec = item_cdf(watch_time_vec)

    def func3(x):
        f = item_cdf_vec - x*watch_time_vec
        return f

    # sol = root(func, 0.3)
    # sol = root(func2, np.random.randn(2), jac=False, method='lm')
    # print(sol.x)
    # sol = root(func3, 0.3, method='lm')

    result, err = integrate.quad(inte_gauss, -np.inf, np.inf, args=(0, 1))


    def func_gaussian(x):
        f = []
        term_sigma = 2 * x[1] * x[1]
        regularization = 1 / np.sqrt(np.pi * term_sigma)
        for item, time in zip(item_cdf_vec, watch_time_vec):
            print(item, time)   # debug
            integrate_result, err = integrate.quad(inte_gauss, -np.inf, time, args=(x[0], x[1]))
            f_item = item - regularization * integrate_result
            print(f_item)   # debug
            f.append(f_item)
        f = np.array(f)
        return f


    sol = root(func_gaussian, np.random.randn(2), method='lm')

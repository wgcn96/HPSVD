# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/7/22'

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import sys

    plt.figure(figsize=(11.5, 8.5))

    # HR
    # 100 missing value
    lfu = [4.21, 5.63, 7.26, 7.58, 7.59]
    lru = [4.47, 6.06, 7.36, 7.56, 7.59]
    mean_popular = [3.83, 4.96, 5.70, 5.78, 5.78]
    popcaching = [7.44, 9.48, 11.43, 11.84, 11.88]
    oplfu = [7.44, 9.48, 11.43, 11.84, 11.88]
    hp_svd = [1.92, 3.58, 7.19, 10.9, 19.56]

    x = [5, 10, 25, 50, 100]
    plt.xlabel('The cache size', fontsize=24)
    plt.ylabel('Hit rate percentage', fontsize=24)

    plt.plot(x, lfu, '#A52A2A', label='LFU', linewidth=3, marker='p', markersize=10, markevery=(4, 5))
    plt.plot(x, lru, '#FF69B4', label='LRU', linewidth=3, marker='h', markersize=10, markevery=(4, 5))
    plt.plot(x, mean_popular, 'g', label='Mean-popular', linewidth=3, marker='s', markersize=10, markevery=(4, 5))
    plt.plot(x, popcaching, 'b', label='Popcaching', linewidth=3, marker='<', markersize=10, markevery=(4, 5))
    plt.plot(x, oplfu, 'c', label='OPLFU', linewidth=3, marker='o', markersize=10, markevery=(4, 5))
    # plt.plot(x, hp_svd, '#D56F2B', label='MLP', linewidth=3, marker='D', markersize=10, markevery=(4, 5))
    plt.plot(x, hp_svd, 'r', label='HP-SVD', linewidth=3, marker='v', markersize=10, markevery=(4, 5))

    plt.ylim(ymin=0, ymax=25)
    plt.xticks(np.arange(0, 110, 10), rotation=0, fontsize=24)
    plt.yticks(np.arange(0, 30, 5), fontsize=24)

    plt.legend(fontsize=24, loc=2, ncol=2, framealpha=0)
    plt.grid()
    plt.savefig('output/person')
    plt.show()

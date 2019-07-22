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
    lfu = [0.62, 1.57, 4.30, 8.47, 15.50]
    lru = [1.34, 2.49, 4.47, 8.12, 13.92]
    mean_popular = [1.69, 3.12, 7.23, 14.61, 23.20]
    popcaching = [1.29, 1.54, 3.08, 4.35, 6.53]
    oplfu = [1.29, 1.54, 3.08, 4.35, 6.53]
    hp_svd = [2.18, 3.45, 8.14, 16.43, 24.86]

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
    plt.savefig('output/province')
    plt.show()

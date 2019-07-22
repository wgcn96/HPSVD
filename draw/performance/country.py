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
    lfu = [0.41, 0.76, 3.21, 6.76, 12.27]
    lru = [0.60, 1.44, 4.95, 8.66, 12.36]
    mean_popular = [1.63, 2.99, 6.75, 14.41, 23.02]
    popcaching = [1.45, 1.74, 2.59, 3.95, 5.32]
    oplfu = [1.45, 1.74, 2.59, 3.95, 5.32]
    hp_svd = [1.55, 3.45, 7.77, 17.31, 24.17]

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
    plt.savefig('output/country')
    plt.show()

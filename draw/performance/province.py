# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/7/22'

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoLocator, ScalarFormatter
    import sys

    plt.figure(figsize=(11.5, 13.5))

    # HR
    # 100 missing value
    lfu = [5.03, 10.39, 16.52, 20.53, 23.85, 40.74, 54.81, ]
    lru = [8.96, 15.89, 23.09, 28.44, 33.78, 51.15, 64.13, ]
    mean_popular = [14.01, 22.26, 31.94, 37.90, 42.02, 55.40, 64.61, ]
    popcaching = [4.17, 5.10, 7.20, 9.05, 12.68, 41.93, 62.20, ]
    oplfu = [6.43, 11.38, 13.83, 16.58, 17.61, 28.28, 35.16, ]
    hp_svd = [15.26, 25.04, 34.37, 40.13, 44.01, 56.57, 68.83, ]

    x = [10, 25, 50, 75, 100, 250, 500]
    plt.xlabel('The cache size', fontsize=40)
    plt.ylabel('Hit rate percentage', fontsize=40)

    plt.plot(x, lfu, '#A52A2A', label='LFU', linewidth=4, marker='p', markersize=13, )
    plt.plot(x, lru, '#FF69B4', label='LRU', linewidth=4, marker='h', markersize=13, )
    plt.plot(x, oplfu, 'c', label='OPLFU', linewidth=4, marker='o', markersize=13, )
    plt.plot(x, mean_popular, 'g', label='Mean-popular', linewidth=4, marker='s', markersize=13, )
    plt.plot(x, popcaching, 'b', label='Popcaching', linewidth=4, marker='<', markersize=13, )

    # plt.plot(x, hp_svd, '#D56F2B', label='MLP', linewidth=4, marker='D', markersize=13, )
    plt.plot(x, hp_svd, 'r', label='HP-SVD', linewidth=4, marker='v', markersize=13, )

    plt.ylim(ymin=-5, ymax=90)
    # plt.xlim(xmin=0, )
    plt.xticks(np.array(x), rotation=0, fontsize=36)
    plt.yticks(np.arange(10, 90, 20), fontsize=36)

    ax = plt.axes()
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.xaxis.set_ticks([10, 20, 40, 100, 500])

    # tx0 = 300
    # ty0 = 63
    # tx1 = 100
    # ty1 = 65
    # arrow = plt.annotate('HP-SVD', xy=(tx0,ty0),xytext=(tx1,ty1),arrowprops=dict(facecolor='r', shrink=0.02, width=8, headwidth=12))
    # arrow.set_color('red')
    # arrow.set_size(36)

    plt.legend(fontsize=34, loc=2, ncol=2, framealpha=0)
    plt.grid(axis='y')
    plt.savefig('output/province.eps')
    plt.show()

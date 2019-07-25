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
    lfu = [4.45, 6.89, 16.03, 20.78, 22.66, 37.96, 52.24, ]
    lru = [6.89, 17.31, 22.60, 28.14, 32.19, 49.82, 62.37, ]
    mean_popular = [13.82, 22.08, 32.29, 38.48, 43.03, 57.16, 67.16, ]
    hp_svd = [14.94, 24.01, 33.82, 39.63, 44.56, 56.95, 72.10, ]
    popcaching = [0.09, 0.12, 0.12, 3.20, 4.27, 23.59, 42.86, ]
    oplfu = [5.28, 9.85, 15.68, 15.68, 16.71, 31.71, 40.55, ]

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
    # plt.xlim(xmin=0,)
    plt.xticks(np.arange(0, 600, 100), rotation=0, fontsize=36)
    plt.yticks(np.arange(10, 90, 20), fontsize=36)

    ax = plt.axes()
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.xaxis.set_ticks([10, 20, 40, 100, 500])

    # tx0 = 290
    # ty0 = 63
    # tx1 = 100
    # ty1 = 65
    # arrow = plt.annotate('HP-SVD', xy=(tx0,ty0),xytext=(tx1,ty1),arrowprops=dict(facecolor='r', shrink=0.02, width=8, headwidth=12))
    # arrow.set_color('red')
    # arrow.set_size(36)

    plt.legend(fontsize=34, loc=2, ncol=2, framealpha=0)
    plt.grid(axis='y')
    plt.savefig('output/country.eps')
    plt.show()

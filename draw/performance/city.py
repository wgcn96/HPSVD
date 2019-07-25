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

    lfu = [6.69, 12.30, 17.08, 21.37, 25.31, 40.73, 53.20, ]
    lru = [7.69, 14.50, 22.48, 28.91, 34.31, 50.46, 62.03, ]
    mean_popular = [13.51, 21.42, 30.07, 35.64, 39.49, 51.02, 58.41, ]
    popcaching = [2.17, 3.36, 7.34, 14.45, 22.25, 45.35, 58.65, ]
    oplfu = [7.23, 9.64, 13.88, 16.34, 20.02, 28.86, 34.45, ]
    hp_svd = [14.90, 24.08, 32.93, 38.32, 42.14, 55.40, 66.21, ]

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

    plt.legend(fontsize=34, loc=2, ncol=2, framealpha=0)
    plt.grid(axis='y')

    # tx0 = 163
    # ty0 = 52
    # tx1 = 100
    # ty1 = 58
    # arrow = plt.annotate('HP-SVD', xy=(tx0,ty0),xytext=(tx1,ty1),arrowprops=dict(facecolor='r', shrink=0.02, width=8, headwidth=13))
    # arrow.set_color('red')
    # arrow.set_size(36)

    plt.savefig('output/city.eps')
    plt.show()

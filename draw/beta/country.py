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

    beta0 = [14.94, 24.01, 33.82, 39.63, 44.56, 56.95, 72.10, ]
    beta1 = [14.94, 24.01, 33.82, 39.63, 44.56, 56.95, 72.10, ]
    hp_svd = [14.94, 24.01, 33.82, 39.63, 44.56, 56.95, 72.10, ]

    x = [10, 25, 50, 75, 100, 250, 500]
    plt.xlabel('The cache size', fontsize=40)
    plt.ylabel('Hit rate percentage', fontsize=40)

    plt.plot(x, hp_svd, 'r', label='HP-SVD', linewidth=4, marker='v', markersize=13, )
    plt.plot(x, beta0, '#A52A2A', label='DSE', linewidth=4, marker='p', markersize=13, )
    # plt.plot(x, lru, '#FF69B4', label='LRU', linewidth=4, marker='h', markersize=13, )
    # plt.plot(x, mean_popular, 'g', label='Mean-popular', linewidth=4, marker='s', markersize=13, )
    # plt.plot(x, popcaching, 'b', label='Popcaching', linewidth=4, marker='<', markersize=13, )
    plt.plot(x, beta1, 'c', label='DME', linewidth=4, marker='o', markersize=13, )
    # plt.plot(x, hp_svd, '#D56F2B', label='MLP', linewidth=4, marker='D', markersize=13, )

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

    plt.legend(fontsize=34, loc=2, framealpha=0)
    plt.grid(axis='y')
    plt.savefig('output/beta_country.eps')
    plt.show()

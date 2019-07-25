# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/7/24'

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#
#     fig, ax = plt.subplots()
#     ax.set_xscale('log', basex=5)
#     ax.set_yscale('log', basey=2)
#
#     ax.plot(range(1024))
#     plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoLocator, ScalarFormatter

    Temp=[10,12.5,15,17.5,20,22.5,25,27.5,30,32.5,35,37.5,40,42.5,45,47.5,50]

    I_threshold = [22.376331312083646, 22.773439481450737, 23.440242034972115,
                   23.969920199339803, 24.80014584753161, 25.275728442307503,
                   26.291852943772966, 26.969268640398795, 28.09683889698702,
                   28.952552190706545, 30.325961112054102, 31.488435380923281,
                   33.176033568454699, 34.613872631424236, 36.710165595581906,
                   38.567151879424728, 41.245216030694756]

    fig, ax = plt.subplots()

    ax.set_yscale('log')
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()

    ax.scatter(Temp,I_threshold)

    plt.xlabel('$ T_c \ (^\circ C)$')
    plt.ylabel('$ I_t \ (mA) $')

    plt.grid(True,which="major",color='black',ls="-",linewidth=0.5)

    plt.show()

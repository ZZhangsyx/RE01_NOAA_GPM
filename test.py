# coding=utf-8

"""
Author: Zhang Zhi
Email: eng.zz@qq.com
Date: 2021/1/10 
Utilize: 
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap

in_path = '../data'
Figname = 'RMSE'
Basin_FH = [35, 39.5, 110, 114]
Basin_DJ = [22.5, 25.5, 113.5, 116]
Scale = 0.1

# load the score data
score = np.load(os.path.join(in_path, 'processed_FH', 'Score.npy'))
lat_bs = np.arange(Basin_FH[0] + Scale / 2, Basin_FH[1], Scale)
lon_bs = np.arange(Basin_FH[2] + Scale / 2, Basin_FH[3], Scale)

labels = ['SVM', 'RF', 'ANN', 'XGB', 'GPM-E']
fontsize = 22
axissize = 22
# cmap = plt.get_cmap('tab20b')
cmap = ListedColormap(['white', 'aquamarine', 'cyan', 'lime', 'limegreen',
                       'yellow', 'gold', 'darkorange', 'r', 'firebrick'])

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(5, 5), sharey=True, sharex=False)

if Figname == 'RMSE':

    data = score[::-1, :, 0, :] / 20
    plt.yticks(np.arange(35.05, 40, 1), ['35N', '36N', '37N', '38N', '39N'])
    plt.xticks(np.arange(110.95, 113, 1), ['111', '112', '113'])

    levels = MaxNLocator(nbins=11).tick_values(0, 1)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    for j in range(5):
        data1 = data[:, :, j]
        im = axes[j].pcolormesh(lon_bs, lat_bs, data1, cmap=cmap, norm=norm)
        axes[j].xaxis.set_tick_params(labelsize=axissize)
        axes[j].yaxis.set_tick_params(labelsize=axissize)
        axes[j].set_aspect('equal', adjustable='box')
        axes[j].grid(linestyle='-.', c='g')
        axes[j].text(110.1, 39.1, 'a' + str(j + 1) + ' ' + labels[j], ha='left', fontsize=fontsize,
                     wrap=True)

    # colorbar
    cax = plt.axes([0.1, 0.2, 0.8, 0.03])
    cb = plt.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_ticks(np.arange(0.1, 1.1, 0.1))
    cb.set_ticklabels(['2', '4', '6', '8', '10', '12', '14', '16', '18', 'mm/day'])
    cb.ax.tick_params(labelsize=fontsize)

    fig.subplots_adjust(bottom=0.2, right=0.96, top=0.8, left=0.06, hspace=0.2, wspace=0.02)
    plt.show()

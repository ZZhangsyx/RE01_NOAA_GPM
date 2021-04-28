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
out_path = '../figure'
Figname = 'RMSE(mm/day)'
Basin_FH = [35, 39.5, 110, 114]
Basin_DJ = [22.5, 25.5, 113.5, 116]
Scale = 0.1

# load the score data
score_DJ = np.load(os.path.join(in_path, 'processed_DJ', 'Score.npy'))
score_FH = np.load(os.path.join(in_path, 'processed_FH', 'Score.npy'))

lat_bs_DJ = np.arange(Basin_DJ[0] + Scale / 2, Basin_DJ[1], Scale)
lon_bs_DJ = np.arange(Basin_DJ[2] + Scale / 2, Basin_DJ[3], Scale)
lat_bs_FH = np.arange(Basin_FH[0] + Scale / 2, Basin_FH[1], Scale)
lon_bs_FH = np.arange(Basin_FH[2] + Scale / 2, Basin_FH[3], Scale)

labels = ['SVM', 'RF', 'ANN', 'XGB']
len_labels = len(labels)

fontsize = 22
axissize = 22
cmap = ListedColormap(['white', 'aquamarine', 'cyan', 'lime', 'limegreen',
                       'yellow', 'gold', 'darkorange', 'r', 'firebrick'])

lat_bs = lat_bs_FH
lon_bs = lon_bs_FH

if Figname == 'RMSE(mm/day)':
    data = score_FH[::-1, :, 0, :]
    title_id = 'a'
    tick_min = -20
    tick_max = 0

if Figname == 'FAR':
    data = score_FH[::-1, :, 3, :]
    title_id = 'b'
    tick_min = -1
    tick_max = 0

if Figname == 'F':
    data = score_FH[::-1, :, 5, :]
    title_id = 'c'
    tick_min = 0
    tick_max = 0.5

cb_ticks = np.arange(tick_min, tick_max + 0.0001, (tick_max - tick_min) / 10)
cb_ticklabels = [' ']
cb_ticklabels.extend([str("%.2f" % cb_tick) for cb_tick in cb_ticks[1:-1]])
cb_ticklabels.append(Figname)
levels = MaxNLocator(nbins=11).tick_values(tick_min, tick_max)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
fig, axes = plt.subplots(nrows=1, ncols=len_labels, figsize=(5, 5), sharey=True, sharex=False)
plt.yticks(np.arange(35.05, 40, 1), ['35N', '36N', '37N', '38N', '39N'])
plt.xticks(np.arange(110.95, 113, 1), ['111E', '112E', '113E'])
for j in range(len_labels):
    data1 = data[:, :, j] - data[:, :, 4]
    data1[data1 == 0] = np.nan
    im = axes[j].pcolormesh(lon_bs, lat_bs, data1, cmap=cmap, norm=norm)
    axes[j].xaxis.set_tick_params(labelsize=axissize)
    axes[j].yaxis.set_tick_params(labelsize=axissize)
    axes[j].set_aspect('equal', adjustable='box')
    axes[j].grid(linestyle='-.', c='g')
    title = title_id + str(j + 1) + ') ' + labels[j]
    axes[j].text(110.1, 39.1, title, ha='left', fontsize=fontsize, wrap=True)

# colorbar
cax = plt.axes([0.1, 0.15, 0.8, 0.03])
cb = plt.colorbar(im, cax=cax, orientation='horizontal')
cb.set_ticks(cb_ticks)
cb.set_ticklabels(cb_ticklabels)
cb.ax.tick_params(labelsize=fontsize)

fig.subplots_adjust(bottom=0.2, right=0.96, top=0.8, left=0.06, hspace=0.2, wspace=0.02)
plt.savefig(out_path + '/Score_dif_figre' + Figname)
plt.show()

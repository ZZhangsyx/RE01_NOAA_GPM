# coding=utf-8

"""
Author: Zhang Zhi
Email: eng.zz@qq.com
Date: 2021/1/24 
Utilize: 
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap

in_path = r'I:\RE01_NOAA_GPM\data\dataset_FH\CMPA_FenHe.npy'
out_path = '../figure'
Figname = 'Pre(mm/day)'
Basin_FH = [35, 39.5, 110, 114]
Basin_DJ = [22.5, 25.5, 113.5, 116]
Scale = 0.1

# load the score data
data = np.load(in_path)
data = data[-365:,::-1,:].sum(axis=0)
isFH = np.load(r'I:\RE01_NOAA_GPM\data\isFenhe.npy')
isFH[isFH==0] = np.nan
data = data * isFH[::-1]

lat_bs = np.arange(Basin_FH[0] + Scale / 2, Basin_FH[1], Scale)
lon_bs = np.arange(Basin_FH[2] + Scale / 2, Basin_FH[3], Scale)

fontsize = 22
axissize = 22
cmap = ListedColormap(['white', 'aquamarine', 'cyan', 'lime', 'limegreen',
                       'yellow', 'gold', 'darkorange', 'r', 'firebrick'])

title_id = 'd'
tick_min = 0
tick_max = 1000

cb_ticks = np.arange(tick_min, tick_max + 0.0001, (tick_max - tick_min) / 10)
cb_ticklabels = [' ']
cb_ticklabels.extend([str("% d" % cb_tick) for cb_tick in cb_ticks[1:-1]])
cb_ticklabels.append(Figname)
levels = MaxNLocator(nbins=11).tick_values(tick_min, tick_max)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
plt.yticks(np.arange(35.05, 40, 1), ['35N', '36N', '37N', '38N', '39N'])
plt.xticks(np.arange(110.95, 113, 1), ['111E', '112E', '113E'])


im = axes.pcolormesh(lon_bs, lat_bs, data, cmap=cmap, norm=norm)
axes.xaxis.set_tick_params(labelsize=axissize)
axes.yaxis.set_tick_params(labelsize=axissize)
axes.set_aspect('equal', adjustable='box')
axes.grid(linestyle='-.', c='g')
title = title_id + ') CMPA'
axes.text(110.1, 39.1, title, ha='left', fontsize=fontsize, wrap=True)

# colorbar
cax = plt.axes([0.65, 0.2, 0.03, 0.6])
cb = plt.colorbar(im, cax=cax)
cb.set_ticks(cb_ticks)
cb.set_ticklabels(cb_ticklabels)
cb.ax.tick_params(labelsize=fontsize)

fig.subplots_adjust(bottom=0.2, right=0.96, top=0.8, left=0.06, hspace=0.2, wspace=0.02)
plt.savefig(out_path + '/Score_dif_figre' + Figname)
plt.show()
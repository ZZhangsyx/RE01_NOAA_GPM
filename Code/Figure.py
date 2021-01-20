# coding=utf-8
"""
Author: Zhang Zhi
Email: zhangzh49@mail2.sysu.edu.cn
Date: 2021/1/5
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from Train_regression import Display_Score
import joblib
import xgboost as xgboost
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap


def Evaluate_STA(DEA_path, processed_path, sta_path, out_path, BN):
    """
    :param DEA_path: DEA path
    :param processed_path: Processed path
    :param sta_path: STA path
    :param out_path: Save path
    :param BN: Basin name's abbreviation
    :return: Performance of comparing to STA
    """
    print('Loading data...')
    data_pro = pd.read_csv(os.path.join(DEA_path, 'STA_' + BN + '.csv'))
    data_pro['Time'] = pd.to_datetime(data_pro['Time'])
    data_pro = data_pro.set_index('Time')

    print('Data processing...')
    Test = data_pro['2018']
    station = pd.read_excel(sta_path, header=2)
    Sta_index = Test.columns.values[1:].astype(int)

    Pred = pd.read_csv(os.path.join(processed_path, 'Test_reg' + BN + '.csv'))
    Pred['Time'] = pd.to_datetime(Pred['Time'])
    Pred = Pred.set_index('Time')

    Lat = pd.read_csv(os.path.join(DEA_path, 'Lat.csv'))
    Lat['Time'] = pd.to_datetime(Lat['Time'])
    Lat = Lat.set_index('Time')
    Lat = np.array(Lat['2018'])[:, 0]

    Lon = pd.read_csv(os.path.join(DEA_path, 'Lon.csv'))
    Lon['Time'] = pd.to_datetime(Lon['Time'])
    Lon = Lon.set_index('Time')
    Lon = np.array(Lon['2018'])[:, 0]

    GPM = pd.read_csv(os.path.join(DEA_path, 'GPM.csv'))
    GPM['Time'] = pd.to_datetime(GPM['Time'])
    GPM = GPM.set_index('Time')
    GPM = GPM['2018']

    GPM_F = pd.read_csv(os.path.join(DEA_path, 'GPF_' + BN + '.csv'))
    GPM_F['Time'] = pd.to_datetime(GPM_F['Time'])
    GPM_F = GPM_F.set_index('Time')

    df_results = pd.DataFrame()
    Methods = ['SVM', 'RF', 'ANN', 'XGB', 'GPM', 'GPF']
    Metrics = ['RMSE', 'R', 'POD', 'FAR', 'ACC', 'F']
    Name = []
    for i in Methods:
        for j in Metrics:
            Name.append(i + '-' + j)
    df_results['name'] = Name
    for Si in Sta_index:
        data1 = station.loc[station['区站号'] == Si]
        Si_lat = np.array(data1['纬度'] / 100)
        Si_lon = np.array(data1['经度'] / 100)

        Loss = np.array((Lat - Si_lat) ** 2 + (Lon - Si_lon) ** 2)
        select_index = np.where(Loss == np.min(Loss))[0]
        Loss_F = np.array((GPM_F['Lat'] - Si_lat) ** 2 + (GPM_F['Lon'] - Si_lon) ** 2)
        select_index_F = np.where(Loss_F == np.min(Loss_F))[0]

        Pred_select = Pred.iloc[select_index]
        GPM_select = GPM.iloc[select_index]
        GPM_select_F = GPM_F.iloc[select_index_F]
        Pred_cat = pd.concat([Pred_select.iloc[:, -4:], GPM_select, GPM_select_F.iloc[:, -1], Test[str(Si)]], axis=1,
                             join='inner')

        performance = []
        for i in range(6):
            performance.extend(list(Display_Score(Pred_cat.iloc[:, i], Pred_cat.iloc[:, -1], isShow=False)))
        df_results[str(Si)] = performance

    df_results.set_index('name')
    df_results.to_csv(os.path.join(out_path, 'Results_STA_' + BN + '.csv'), index=True)


def get_score(processed_path, DEA_path, Basin, Scale, Basin_name, BN, BS):
    csv = pd.read_csv(os.path.join(processed_path, 'Test_reg' + BN + '.csv'))
    Lat = pd.read_csv(os.path.join(DEA_path, 'Lat.csv'))
    Lat['Time'] = pd.to_datetime(Lat['Time'])
    Lat = Lat.set_index('Time')
    Lat = np.array(Lat['2018'])[:, 0]

    Lon = pd.read_csv(os.path.join(DEA_path, 'Lon.csv'))
    Lon['Time'] = pd.to_datetime(Lon['Time'])
    Lon = Lon.set_index('Time')
    Lon = np.array(Lon['2018'])[:, 0]

    GPM = pd.read_csv(os.path.join(DEA_path, 'GPM.csv'))
    GPM['Time'] = pd.to_datetime(GPM['Time'])
    GPM = GPM.set_index('Time')
    GPM = np.array(GPM['2018'])

    csv['Lat'], csv['Lon'], csv['GPM_F'] = Lat, Lon, GPM

    lat_bs = np.arange(Basin[0] + Scale / 2, Basin[1], Scale)
    lon_bs = np.arange(Basin[2] + Scale / 2, Basin[3], Scale)
    isBasin = np.load(os.path.join('../data/is' + Basin_name + '.npy'))

    mean_pre = np.zeros([len(lat_bs), len(lon_bs), 6, 5])

    for ilat in np.arange(len(lat_bs)):
        for jlon in np.arange(len(lon_bs)):
            if isBasin[ilat, jlon]:
                lat_grid = lat_bs[ilat]
                lon_grid = lon_bs[jlon]
                select_index = (abs(Lat - lat_grid) <= 0.01) * (abs(Lon - lon_grid) <= 0.01)
                csv_select = csv.iloc[select_index]

                if csv_select.shape[0] != 0:
                    mean_pre[ilat, jlon, :, 0] = Display_Score(csv_select['SVM_cat'], csv_select['CMPA_' + BS])
                    mean_pre[ilat, jlon, :, 1] = Display_Score(csv_select['RF_cat'], csv_select['CMPA_' + BS])
                    mean_pre[ilat, jlon, :, 2] = Display_Score(csv_select['ANN_cat'], csv_select['CMPA_' + BS])
                    mean_pre[ilat, jlon, :, 3] = Display_Score(csv_select['XGB_cat'], csv_select['CMPA_' + BS])
                    mean_pre[ilat, jlon, :, 4] = Display_Score(csv_select['GPM_' + BS], csv_select['CMPA_' + BS])

    mean_pre[np.isnan(mean_pre)] = 0
    np.save(os.path.join(processed_path, 'Score.npy'), mean_pre)


def Boxplot(in_path, out_path, BN):
    """
    :param in_path: Csv path
    :param out_path: Figure save path
    :param BN: Basin name's abbreviation
    :return: Plot and save the boxplot figure
    """
    DATA = pd.read_csv(in_path + '/Results_STA_' + BN + '.csv')
    labels = ['SVM', 'RF', 'ANN', 'XGB', 'GPM-E', 'GPM-F']
    fs = 16  # fontsize
    meanpointprops = dict(marker='D', markeredgecolor='black',
                          markerfacecolor='firebrick')
    medianprops = dict(color='black', linewidth=1.5)
    colors = ['cyan', 'lime', 'limegreen', 'yellow', 'darkorange', 'r']

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=False, sharex=True)
    for i in range(2):
        for j in range(3):
            index = 3 * i + j
            data = []
            for k in range(6):
                data.append(DATA.iloc[index + k * 6, 2:].astype(float).values)

            bplot = axes[i, j].boxplot(data, labels=labels, meanprops=meanpointprops, meanline=False,
                                       showmeans=True, medianprops=medianprops, patch_artist=True)
            axes[i, j].xaxis.set_tick_params(labelsize=fs)
            axes[i, j].yaxis.set_tick_params(labelsize=fs)

            # fill with colors
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

    fig.subplots_adjust(hspace=0.1)
    plt.savefig(out_path + "/Boxplot_" + BN)
    plt.show()


def get_importance(model_path):
    model = joblib.load(model_path)
    importance = model.get_booster().get_score()
    temp1 = []
    temp2 = []
    for k in importance:
        temp1.append(k)
        temp2.append(importance[k])
    df_im = pd.DataFrame({
        'column': temp1,
        'importance': temp2,
    }).sort_values(by='importance', ascending=False)
    df_im = df_im.set_index('column')
    df_im = df_im.drop(['Lat', 'Lon'])
    return df_im


def ImportancePlot(model_path, out_path):
    model_name = ['XGB_C_DJ', 'XGB_R_DJ', 'XGB_C_FH', 'XGB_R_FH']
    title_name = ['1) Classifier in DJR', '2) Regressor in DJR', '3) Classifier in FHR', '4) Regressor in FHR']
    feature_name = [
        ['', 'TC-mean', 'TC-max', 'CH-max', 'T375-mean', 'R375-mean', 'R160-max', 'T375-max', 'STR-mean',
         'GPM', 'R065-mean'],
        ['', 'GPM', 'CH-max', 'T375-mean', 'TC-mean', 'CTr-mean', 'T375-max', 'STR-mean', 'TC-max',
         'CR-A-mean', 'R160-max'],
        ['', 'TC-mean', 'GPM', 'CH-max', 'T375-mean', 'TC-max', 'R160-max', 'CTr-mean', 'R065-max',
         'STR-max', 'ST'],
        ['', 'GPM', 'CH-max', 'STR-mean', 'R065-mean', 'CTr-mean', 'TC-mean', 'CH-mean', 'TC-max',
         'R160-max', 'CTe-max']]

    fs = 18  # fontsize
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6), sharey=False, sharex=True)
    for i in range(2):
        for j in range(2):
            index = 2 * i + j
            importance = get_importance(os.path.join(model_path, model_name[index] + '.pkl'))
            performance = importance.iloc[:, 0]
            performance = performance / performance.sum()
            performance = performance[:10]

            if index < 2:
                axes[i, j].barh(np.arange(1, 11), performance, color='firebrick')
            else:
                axes[i, j].barh(np.arange(1, 11), performance, color='goldenrod')

            axes[i, j].invert_yaxis()  # labels read top-to-bottom
            axes[i, j].set_yticks(np.arange(11))
            axes[i, j].set_yticklabels(feature_name[index])
            axes[i, j].text(0.07, 10.2, title_name[index], fontsize=fs)
            axes[i, j].xaxis.set_tick_params(labelsize=18)
            axes[i, j].yaxis.set_tick_params(labelsize=18)

            if index > 1:
                axes[i, j].set_xlabel('Importance', fontsize=fs)

    fig.subplots_adjust(hspace=0.1, wspace=0.25)
    plt.savefig(out_path + "/Importance.png")
    plt.show()


def ScorePlot(in_path, out_path, Basin_DJ, Basin_FH, Scale, Figname):
    # load the score data
    score_DJ = np.load(os.path.join(in_path, 'processed_DJ', 'Score.npy'))
    score_FH = np.load(os.path.join(in_path, 'processed_FH', 'Score.npy'))

    lat_bs_DJ = np.arange(Basin_DJ[0] + Scale / 2, Basin_DJ[1], Scale)
    lon_bs_DJ = np.arange(Basin_DJ[2] + Scale / 2, Basin_DJ[3], Scale)
    lat_bs_FH = np.arange(Basin_FH[0] + Scale / 2, Basin_FH[1], Scale)
    lon_bs_FH = np.arange(Basin_FH[2] + Scale / 2, Basin_FH[3], Scale)

    labels = ['SVM', 'RF', 'ANN', 'XGB', 'GPM-E']
    fontsize = 22
    axissize = 22
    cmap = ListedColormap(['white', 'aquamarine', 'cyan', 'lime', 'limegreen',
                           'yellow', 'gold', 'darkorange', 'r', 'firebrick'])
    levels = MaxNLocator(nbins=11).tick_values(0, 1)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    if Figname == 'RMSE':
        # load the score data
        lat_bs = lat_bs_FH
        lon_bs = lon_bs_FH
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(5, 5), sharey=True, sharex=False)

        data = score_FH[::-1, :, 0, :] / 20
        plt.yticks(np.arange(35.05, 40, 1), ['35N', '36N', '37N', '38N', '39N'])
        plt.xticks(np.arange(110.95, 113, 1), ['111', '112', '113'])

        for j in range(5):
            data1 = data[:, :, j]
            im = axes[j].pcolormesh(lon_bs, lat_bs, data1, cmap=cmap, norm=norm)
            axes[j].xaxis.set_tick_params(labelsize=axissize)
            axes[j].yaxis.set_tick_params(labelsize=axissize)
            axes[j].set_aspect('equal', adjustable='box')
            axes[j].grid(linestyle='-.', c='g')
            axes[j].text(110.1, 39.1, 'a' + str(j + 1) + ') ' + labels[j], ha='left', fontsize=fontsize,
                         wrap=True)

        # colorbar
        cax = plt.axes([0.1, 0.2, 0.8, 0.03])
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        cb.set_ticks(np.arange(0.1, 1.1, 0.1))
        cb.set_ticklabels(['2', '4', '6', '8', '10', '12', '14', '16', '18', 'mm/day'])
        cb.ax.tick_params(labelsize=fontsize)

        fig.subplots_adjust(bottom=0.2, right=0.96, top=0.8, left=0.06, hspace=0.2, wspace=0.02)
        plt.savefig(out_path + '/Score_figre' + Figname)
        plt.show()

    if Figname == 'FAR':
        # load the score data
        lat_bs = lat_bs_FH
        lon_bs = lon_bs_FH
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(5, 5), sharey=True, sharex=False)

        data = score_FH[::-1, :, 3, :]
        plt.yticks(np.arange(35.05, 40, 1), ['35N', '36N', '37N', '38N', '39N'])
        plt.xticks(np.arange(110.95, 113, 1), ['111', '112', '113'])

        for j in range(5):
            data1 = data[:, :, j]
            im = axes[j].pcolormesh(lon_bs, lat_bs, data1, cmap=cmap, norm=norm)
            axes[j].xaxis.set_tick_params(labelsize=axissize)
            axes[j].yaxis.set_tick_params(labelsize=axissize)
            axes[j].set_aspect('equal', adjustable='box')
            axes[j].grid(linestyle='-.', c='g')
            axes[j].text(110.1, 39.1, 'b' + str(j + 1) + ') ' + labels[j], ha='left', fontsize=fontsize,
                         wrap=True)

        # colorbar
        cax = plt.axes([0.1, 0.2, 0.8, 0.03])
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        cb.set_ticks(np.arange(0.1, 1.1, 0.1))
        cb.set_ticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', ''])
        cb.ax.tick_params(labelsize=fontsize)

        fig.subplots_adjust(bottom=0.2, right=0.96, top=0.8, left=0.06, hspace=0.2, wspace=0.02)
        plt.savefig(out_path + '/Score_figre' + Figname)
        plt.show()

    if Figname == 'F':
        # load the score data
        lat_bs = lat_bs_FH
        lon_bs = lon_bs_FH
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(5, 5), sharey=True, sharex=False)

        data = score_FH[::-1, :, 5, :]
        plt.yticks(np.arange(35.05, 40, 1), ['35N', '36N', '37N', '38N', '39N'])
        plt.xticks(np.arange(110.95, 113, 1), ['111', '112', '113'])

        for j in range(5):
            data1 = data[:, :, j]
            im = axes[j].pcolormesh(lon_bs, lat_bs, data1, cmap=cmap, norm=norm)
            axes[j].xaxis.set_tick_params(labelsize=axissize)
            axes[j].yaxis.set_tick_params(labelsize=axissize)
            axes[j].set_aspect('equal', adjustable='box')
            axes[j].grid(linestyle='-.', c='g')
            axes[j].text(110.1, 39.1, 'c' + str(j + 1) + ') ' + labels[j], ha='left', fontsize=fontsize,
                         wrap=True)

        # colorbar
        cax = plt.axes([0.1, 0.2, 0.8, 0.03])
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        cb.set_ticks(np.arange(0.1, 1.1, 0.1))
        cb.set_ticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', ''])
        cb.ax.tick_params(labelsize=fontsize)

        fig.subplots_adjust(bottom=0.2, right=0.96, top=0.8, left=0.06, hspace=0.2, wspace=0.02)
        plt.savefig(out_path + '/Score_figre' + Figname)
        plt.show()

    if Figname == 0:
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(4, 4), sharey=True, sharex=True)
        for i in range(2):
            data = score_DJ[::-1, :, i, :]
            lon_bs = lon_bs_DJ
            lat_bs = lat_bs_DJ
            if i == 0:
                data = data / 20
            levels = MaxNLocator(nbins=11).tick_values(0, 1)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            for j in range(5):
                index = 5 * i + j

                data1 = data[:, :, j]
                im = axes[i, j].pcolormesh(lon_bs, lat_bs, data1, cmap=cmap, norm=norm)
                axes[i, j].xaxis.set_tick_params(labelsize=18)
                axes[i, j].yaxis.set_tick_params(labelsize=18)
                axes[i, j].set_aspect('equal', adjustable='box')
                axes[i, j].grid(linestyle='-.', c='g')
                if index == 0:
                    axes[i, j].set_ylabel('RMSE', fontsize=22)
                if index == 5:
                    axes[i, j].set_ylabel('R', fontsize=22)

                if index < 5:
                    axes[i, j].set_title(labels[j], fontsize=22)
                    axes[i, j].text(113.6, 25.1, 'a' + str(j + 1), ha='left', fontsize=22, wrap=True)
                else:
                    axes[i, j].text(113.6, 25.1, 'b' + str(j + 1), ha='left', fontsize=22, wrap=True)

        plt.yticks(np.arange(23, 26, 0.5), ['23N', '23.5N', '24N', '24.5N', '25N', '25.5N'])
        plt.xticks(np.arange(113.55, 116, 0.5), ['113.5E', '', '114.5E', '', '115.5E'])
        plt.subplots_adjust(bottom=0.2, right=0.88, top=0.8, left=0.05)
        cax = plt.axes([0.89, 0.2, 0.03, 0.6])
        cb = plt.colorbar(im, cax=cax)
        cb.set_ticks(np.arange(0.1, 1.1, 0.1))
        cb.set_ticklabels(['2(0.1)', '4(0.2)', '6(0.3)', '8(0.4)', '10(0.5)',
                           '12(0.6)', '14(0.7)', '16(0.8)', '18(0.9)', 'RMSE(R)'])
        cb.ax.tick_params(labelsize=18)
        fig.subplots_adjust(hspace=0.1, wspace=0.02)
        plt.savefig(out_path + '/Score_figre' + str(Figname))
        plt.show()

    if Figname == 1:
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(4, 4), sharey=True, sharex=True)
        for i in range(2):
            data = score_DJ[::-1, :, i + 2, :]
            lon_bs = lon_bs_DJ
            lat_bs = lat_bs_DJ
            if i == 1:
                data = data + 0.3
            levels = MaxNLocator(nbins=11).tick_values(0.5, 1)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            for j in range(5):
                index = 5 * i + j

                data1 = data[:, :, j]
                im = axes[i, j].pcolormesh(lon_bs, lat_bs, data1, cmap=cmap, norm=norm)
                axes[i, j].xaxis.set_tick_params(labelsize=18)
                axes[i, j].yaxis.set_tick_params(labelsize=18)
                axes[i, j].set_aspect('equal', adjustable='box')
                axes[i, j].grid(linestyle='-.', c='g')
                if index == 0:
                    axes[i, j].set_ylabel('POD', fontsize=22)
                if index == 5:
                    axes[i, j].set_ylabel('FAR', fontsize=22)

                if index < 5:
                    axes[i, j].set_title(labels[j], fontsize=fontsize)
                    axes[i, j].text(113.6, 25.1, 'c' + str(j + 1), ha='left', fontsize=20, wrap=True)
                else:
                    axes[i, j].text(113.6, 25.1, 'd' + str(j + 1), ha='left', fontsize=20, wrap=True)

        plt.yticks(np.arange(23, 26, 0.5), ['23N', '23.5N', '24N', '24.5N', '25N', '25.5N'])
        plt.xticks(np.arange(113.55, 116, 0.5), ['113.5E', '', '114.5E', '', '115.5E'])
        plt.subplots_adjust(bottom=0.2, right=0.88, top=0.8, left=0.05)
        cax = plt.axes([0.89, 0.2, 0.03, 0.6])
        cb = plt.colorbar(im, cax=cax)
        cb.set_ticks(np.arange(0.5, 1.05, 0.05))
        cb.set_ticklabels(['0.5(0.2)', '0.55(0.25)', '0.6(0.3)', '0.65(0.35)', '0.7(0.4)',
                           '0.75(0.45)', '0.8(0.5)', '0.85(0.55)', '0.9(0.6)', '0.95(0.65)', 'POD(FAR)'])
        cb.ax.tick_params(labelsize=18)
        fig.subplots_adjust(hspace=0.1, wspace=0.02)
        plt.savefig(out_path + '/Score_figre' + str(Figname))
        plt.show()

    if Figname == 2:
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(4, 4), sharey=True, sharex=True)
        for i in range(2):
            data = score_DJ[::-1, :, i + 4, :]
            lon_bs = lon_bs_DJ
            lat_bs = lat_bs_DJ
            if i == 1:
                data = data + 0
            levels = MaxNLocator(nbins=13).tick_values(0.4, 1)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            for j in range(5):
                index = 5 * i + j

                data1 = data[:, :, j]
                im = axes[i, j].pcolormesh(lon_bs, lat_bs, data1, cmap=cmap, norm=norm)
                axes[i, j].xaxis.set_tick_params(labelsize=18)
                axes[i, j].yaxis.set_tick_params(labelsize=18)
                axes[i, j].set_aspect('equal', adjustable='box')
                axes[i, j].grid(linestyle='-.', c='g')
                if index == 0:
                    axes[i, j].set_ylabel('ACC', fontsize=22)
                if index == 5:
                    axes[i, j].set_ylabel('F', fontsize=22)

                if index < 5:
                    axes[i, j].set_title(labels[j], fontsize=fontsize)
                    axes[i, j].text(113.6, 25.1, 'e' + str(j + 1), ha='left', fontsize=20, wrap=True)
                else:
                    axes[i, j].text(113.6, 25.1, 'f' + str(j + 1), ha='left', fontsize=20, wrap=True)

        plt.yticks(np.arange(23, 26, 0.5), ['23N', '23.5N', '24N', '24.5N', '25N', '25.5N'])
        plt.xticks(np.arange(113.55, 116, 0.5), ['113.5E', '', '114.5E', '', '115.5E'])
        plt.subplots_adjust(bottom=0.2, right=0.88, top=0.8, left=0.05)
        cax = plt.axes([0.89, 0.2, 0.03, 0.6])
        cb = plt.colorbar(im, cax=cax)
        cb.set_ticks(np.arange(0.4, 1.05, 0.05))
        cb.ax.tick_params(labelsize=18)
        fig.subplots_adjust(hspace=0.1, wspace=0.02)
        plt.savefig(out_path + '/Score_figre' + str(Figname))
        plt.show()

    if Figname == 3:
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(4, 4), sharey=True, sharex=True)
        for i in range(2):
            data = score_FH[::-1, :, i, :]
            lon_bs = lon_bs_FH
            lat_bs = lat_bs_FH
            if i == 0:
                data = data / 15
            levels = MaxNLocator(nbins=11).tick_values(0, 1)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            for j in range(5):
                index = 5 * i + j

                data1 = data[:, :, j]
                im = axes[i, j].pcolormesh(lon_bs, lat_bs, data1, cmap=cmap, norm=norm)
                axes[i, j].xaxis.set_tick_params(labelsize=18)
                axes[i, j].yaxis.set_tick_params(labelsize=18)
                axes[i, j].set_aspect('equal', adjustable='box')
                axes[i, j].grid(linestyle='-.', c='g')
                if index == 0:
                    axes[i, j].set_ylabel('RMSE', fontsize=22)
                if index == 5:
                    axes[i, j].set_ylabel('R', fontsize=22)

                if index < 5:
                    axes[i, j].set_title(labels[j], fontsize=fontsize)
                    axes[i, j].text(110.1, 39.1, 'a' + str(j + 1), ha='left', fontsize=20, wrap=True)
                else:
                    axes[i, j].text(110.1, 39.1, 'b' + str(j + 1), ha='left', fontsize=20, wrap=True)

        plt.yticks(np.arange(35.05, 40, 1), ['35N', '36N', '37N', '38N', '39N'])
        plt.xticks(np.arange(110.95, 113, 1), ['111E', '112E', '113E'])
        plt.subplots_adjust(bottom=0.2, right=0.88, top=0.8, left=0.05)
        cax = plt.axes([0.9, 0.2, 0.03, 0.6])
        cb = fig.colorbar(im, ax=axes[:, :], cax=cax)
        cb.set_ticks(np.arange(0.1, 1.1, 0.1))
        cb.set_ticklabels(['1.5(0.1)', '3   (0.2)', '4.5(0.3)', '6   (0.4)', '7.5(0.5)',
                           '9   (0.6)', '10.5(0.7)', '12   (0.8)', '13.5(0.9)', 'RMSE(R)'])
        cb.ax.tick_params(labelsize=18)
        fig.subplots_adjust(hspace=0.1)
        plt.savefig(out_path + '/Score_figre' + str(Figname))
        plt.show()

    if Figname == 4:
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(4, 4), sharey=True, sharex=True)
        for i in range(2):
            data = score_FH[::-1, :, i + 2, :]
            lon_bs = lon_bs_FH
            lat_bs = lat_bs_FH
            levels = MaxNLocator(nbins=11).tick_values(0, 1)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            for j in range(5):
                index = 5 * i + j

                data1 = data[:, :, j]
                im = axes[i, j].pcolormesh(lon_bs, lat_bs, data1, cmap=cmap, norm=norm)
                axes[i, j].xaxis.set_tick_params(labelsize=18)
                axes[i, j].yaxis.set_tick_params(labelsize=18)
                axes[i, j].set_aspect('equal', adjustable='box')
                axes[i, j].grid(linestyle='-.', c='g')
                if index == 0:
                    axes[i, j].set_ylabel('POD', fontsize=22)
                if index == 5:
                    axes[i, j].set_ylabel('FAR', fontsize=22)

                if index < 5:
                    axes[i, j].set_title(labels[j], fontsize=fontsize)
                    axes[i, j].text(110.1, 39.1, 'c' + str(j + 1), ha='left', fontsize=20, wrap=True)
                else:
                    axes[i, j].text(110.1, 39.1, 'd' + str(j + 1), ha='left', fontsize=20, wrap=True)

        plt.yticks(np.arange(35.05, 40, 1), ['35N', '36N', '37N', '38N', '39N'])
        plt.xticks(np.arange(110.95, 113, 1), ['111E', '112E', '113E'])
        plt.subplots_adjust(bottom=0.2, right=0.88, top=0.8, left=0.05)
        cax = plt.axes([0.89, 0.2, 0.03, 0.6])
        cb = plt.colorbar(im, cax=cax)
        cb.set_ticks(np.arange(0, 1.05, 0.1))
        cb.ax.tick_params(labelsize=18)
        fig.subplots_adjust(hspace=0.1, wspace=0.02)
        plt.savefig(out_path + '/Score_figre' + str(Figname))
        plt.show()

    if Figname == 5:
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(4, 4), sharey=True, sharex=True)
        for i in range(2):
            data = score_FH[::-1, :, i + 4, :]
            lon_bs = lon_bs_FH
            lat_bs = lat_bs_FH
            levels = MaxNLocator(nbins=11).tick_values(0, 1)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            for j in range(5):
                index = 5 * i + j

                data1 = data[:, :, j]
                im = axes[i, j].pcolormesh(lon_bs, lat_bs, data1, cmap=cmap, norm=norm)
                axes[i, j].xaxis.set_tick_params(labelsize=18)
                axes[i, j].yaxis.set_tick_params(labelsize=18)
                axes[i, j].set_aspect('equal', adjustable='box')
                axes[i, j].grid(linestyle='-.', c='g')
                if index == 0:
                    axes[i, j].set_ylabel('ACC', fontsize=22)
                if index == 5:
                    axes[i, j].set_ylabel('F', fontsize=22)

                if index < 5:
                    axes[i, j].set_title(labels[j], fontsize=fontsize)
                    axes[i, j].text(110.1, 39.1, 'e' + str(j + 1), ha='left', fontsize=20, wrap=True)
                else:
                    axes[i, j].text(110.1, 39.1, 'f' + str(j + 1), ha='left', fontsize=20, wrap=True)

        plt.yticks(np.arange(35.05, 40, 1), ['35N', '36N', '37N', '38N', '39N'])
        plt.xticks(np.arange(110.95, 113, 1), ['111E', '112E', '113E'])
        plt.subplots_adjust(bottom=0.2, right=0.88, top=0.8, left=0.05)
        cax = plt.axes([0.89, 0.2, 0.03, 0.6])
        cb = plt.colorbar(im, cax=cax)
        cb.set_ticks(np.arange(0, 1.05, 0.1))
        cb.ax.tick_params(labelsize=18)
        fig.subplots_adjust(hspace=0.1, wspace=0.02)
        plt.savefig(out_path + '/Score_figre' + str(Figname))
        plt.show()


def DistributionPlot(out_path):
    axissize = 22
    DJ = np.array(pd.read_csv('../data/processed_DJ/DEA/CMPA.csv'))
    FH = np.array(pd.read_csv('../data/processed_FH/DEA/CMPA.csv'))

    DJ_NR = np.sum(DJ[:, 1] == 0)
    FH_NR = np.sum(FH[:, 1] == 0)
    DJ_R = DJ.size / 2 - DJ_NR
    FH_R = FH.size / 2 - FH_NR

    print('Figure...')
    feature_name = ['DJR Rain', 'DJR No Rain', 'FHR Rain', 'FHR No Rain']

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9), sharey=False)

    axes.barh(np.arange(4), [DJ_R, DJ_NR, FH_R, FH_NR], color=['firebrick', 'firebrick', 'goldenrod', 'goldenrod'],
              alpha=0.8)
    axes.set_yticks(np.arange(4))
    axes.set_yticklabels(feature_name, fontsize=axissize)
    axes.invert_yaxis()  # labels read top-to-bottom
    axes.xaxis.set_tick_params(labelsize=axissize)
    axes.yaxis.set_tick_params(labelsize=axissize)
    axes.set_xlabel('Numbers', fontsize=axissize)

    fig.subplots_adjust(hspace=0.3)
    plt.savefig(out_path + '/Distribution.png')
    plt.show()


def TimeseriesPlot(in_path, processed_path, sta_path, out_path, BN, Si):
    csv_name = os.path.join(in_path, 'STA_' + BN + '.csv')
    data_pro = pd.read_csv(csv_name)
    data_pro['Time'] = pd.to_datetime(data_pro['Time'])
    data_pro = data_pro.set_index('Time')

    Test = data_pro['2018']
    station = pd.read_excel(sta_path, header=2)

    Pred = pd.read_csv(os.path.join(processed_path, 'Test_reg' + BN + '.csv'))
    Pred['Time'] = pd.to_datetime(Pred['Time'])
    Pred = Pred.set_index('Time')

    Lat = pd.read_csv(os.path.join(in_path, 'Lat.csv'))
    Lat['Time'] = pd.to_datetime(Lat['Time'])
    Lat = Lat.set_index('Time')
    Lat = np.array(Lat['2018'])[:, 0]

    Lon = pd.read_csv(os.path.join(in_path, 'Lon.csv'))
    Lon['Time'] = pd.to_datetime(Lon['Time'])
    Lon = Lon.set_index('Time')
    Lon = np.array(Lon['2018'])[:, 0]

    GPM = pd.read_csv(os.path.join(in_path, 'GPM.csv'))
    GPM['Time'] = pd.to_datetime(GPM['Time'])
    GPM = GPM.set_index('Time')
    GPM = GPM['2018']

    GPM_F = pd.read_csv(os.path.join(in_path, 'GPF_' + BN + '.csv'))
    GPM_F['Time'] = pd.to_datetime(GPM_F['Time'])
    GPM_F = GPM_F.set_index('Time')

    data1 = station.loc[station['区站号'] == Si]
    Si_lat = np.array(data1['纬度'] / 100)
    Si_lon = np.array(data1['经度'] / 100)

    Loss = np.array((Lat - Si_lat) ** 2 + (Lon - Si_lon) ** 2)
    select_index = np.where(Loss == np.min(Loss))[0]
    Loss_F = np.array((GPM_F['Lat'] - Si_lat) ** 2 + (GPM_F['Lon'] - Si_lon) ** 2)
    select_index_F = np.where(Loss_F == np.min(Loss_F))[0]

    Pred_select = Pred.iloc[select_index]
    GPM_select = GPM.iloc[select_index]
    GPM_select_F = GPM_F.iloc[select_index_F]
    DATA = pd.concat([Pred_select.iloc[:, -4:], GPM_select, GPM_select_F.iloc[:, -1], Test[str(Si)]], axis=1,
                     join='inner')

    labels = ['SVM', 'RF', 'ANN', 'XGB', 'GPM-E', 'GPM-F', 'STA']
    fs = 16  # fontsize
    colors = ['cyan', 'lime', 'limegreen', 'yellow', 'darkorange', 'r', 'gray']
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True, sharex=True)

    for i in range(2):
        for j in range(3):
            index = 3 * i + j
            Y = DATA.iloc[:, -1]
            X = DATA.iloc[:, index]
            performance = Display_Score(X, Y, isShow=False)[1:]
            t = (str(index + 1) + ') ' + labels[index] + '\n' +
                 "RMSE = " + str('%.3f' % performance[0]) + ' mm/day\n' +
                 "R   = " + str('%.3f' % performance[1]) + '\n' +
                 "POD = " + str('%.3f' % performance[2]) + '\n' +
                 "FAR = " + str('%.3f' % performance[3]) + '\n' +
                 "ACC = " + str('%.3f' % performance[4]) + '\n' +
                 "F   = " + str('%.3f' % performance[5]) + '\n')

            axes[i, j].plot(range(len(Y)), Y, label=labels[-1], c=colors[-1])
            axes[i, j].plot(range(len(X)), X, label=labels[index], c=colors[index])
            axes[i, j].fill_between(range(len(X)), np.zeros(len(X)), Y, facecolor='darkgrey', alpha=0.7,
                                    interpolate=True)
            axes[i, j].text(5, 45, t, ha='left', fontsize=16, wrap=True)
            axes[i, j].xaxis.set_tick_params(labelsize=16)
            axes[i, j].yaxis.set_tick_params(labelsize=16)
            axes[i, j].legend(loc='upper right', fontsize=16)
            plt.ylim([0, 100])
            plt.xlim([0, len(X)])

            if index >= 3:
                axes[i, j].xaxis.set_ticks(np.arange(0, 300, 50))
                axes[i, j].xaxis.set_ticklabels(['1-1', '3-1', '5-1', '7-1', '9-1', '11-1'])

    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig(out_path + '/Timeseries.png')
    plt.show()


def Statisticplot():
    XGB_DJ = [0.770, 0.806, 0.370, 0.707, 6.773 / 20, 0.619]
    GPM_DJ = [0.670, 0.869, 0.487, 0.645, 15.753 / 20, 0.598]
    XGB_FH = np.negative([0.861, 0.719, 0.377, 0.667, 3.757 / 20, 0.582])
    GPM_FH = np.negative([0.524, 0.884, 0.726, 0.41, 11.129 / 20, 0.414])

    fontsize = 18
    axissize = 22
    n_groups = 6
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, GPM_DJ, bar_width,
                     alpha=opacity,
                     color='firebrick',
                     error_kw=error_config,
                     label='GPM-E DJ')

    rects2 = plt.bar(index + bar_width, XGB_DJ, bar_width,
                     alpha=opacity,
                     color='goldenrod',
                     error_kw=error_config,
                     label='XGB DJ')
    rects3 = plt.bar(index, GPM_FH, bar_width,
                     alpha=opacity / 2,
                     color='firebrick',
                     error_kw=error_config,
                     label='GPM-E FH')

    rects4 = plt.bar(index + bar_width, XGB_FH, bar_width,
                     alpha=opacity / 2,
                     color='goldenrod',
                     error_kw=error_config,
                     label='XGB FH')

    XGB_DJ = [0.770, 0.806, 0.370, 0.707, 6.773, 0.619]
    GPM_DJ = [0.670, 0.869, 0.487, 0.645, 15.753, 0.598]
    XGB_FH = [0.861, 0.719, 0.377, 0.667, 3.757, 0.582]
    GPM_FH = [0.524, 0.884, 0.726, 0.41, 11.129, 0.414]
    ind = 0
    for rects, data in zip([rects1, rects2, rects3, rects4], [GPM_DJ, XGB_DJ, GPM_FH, XGB_FH]):
        for rect, h in zip(rects, data):
            if ind < 2:
                va = 'bottom'
            else:
                va = 'top'
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height,
                     str('%.3f' % h), size=fontsize, ha='center', va=va)
        ind += 1
    plt.text(5.8, 0.2, 'DJR', size=axissize)
    plt.text(5.8, -0.2, 'FHR', size=axissize)
    plt.plot([-1, 7], [0, 0], 'k')
    plt.xlim([-0.5, 6.5])
    plt.ylim([-1, 1])
    plt.yticks(np.linspace(-1, 1, 11),
               ['1', '0.8', '0.6', '0.4', '0.2', '0', '0.2', '0.4', '0.6', '0.8', '1'],
               fontsize=axissize)
    plt.xticks(index + bar_width / 2, ('ACC', 'POD', 'FAR', 'F', 'RMSE', 'R'), fontsize=axissize)
    plt.rcParams.update({'font.size': fontsize - 2})
    plt.legend()

    plt.tight_layout()
    plt.show()


def Statisticplot_reg():
    SVM_REG = np.array([0.507, 0.961, 0.591, 0.574, 5.904 / 10, 0.647])
    SVM_FINAL = np.array([0.754, 0.793, 0.390, 0.690, 5.909 / 10, 0.647])
    RF_REG = np.array([0.346, 1, 0.655, 0.513, 6.408 / 10, 0.633])
    RF_FINAL = np.array([0.760, 0.813, 0.384, 0.701, 6.359 / 10, 0.632])
    ANN_REG = np.array([0.525, 0.957, 0.582, 0.582, 6.878 / 10, 0.588])
    ANN_FINAL = np.array([0.741, 0.771, 0.403, 0.673, 6.831 / 10, 0.591])
    XGB_REG = np.array([0.479, 0.968, 0.604, 0.562, 6.856 / 10, 0.612])
    XGB_FINAL = np.array([0.770, 0.806, 0.370, 0.707, 6.773 / 10, 0.619])

    fontsize = 18
    axissize = 22
    n_groups = 6
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.15
    opacity = 1
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, SVM_REG, bar_width,
                     alpha=opacity / 2,
                     color='cyan',
                     error_kw=error_config)
    rects2 = plt.bar(index + bar_width, RF_REG, bar_width,
                     alpha=opacity / 2,
                     color='lime',
                     error_kw=error_config)
    rects3 = plt.bar(index + 2 * bar_width, ANN_REG, bar_width,
                     alpha=opacity / 2,
                     color='yellow',
                     error_kw=error_config)
    rects4 = plt.bar(index + 3 * bar_width, XGB_REG, bar_width,
                     alpha=opacity / 2,
                     color='firebrick',
                     error_kw=error_config)

    rects5 = plt.bar(index, SVM_FINAL - SVM_REG, bar_width, bottom=SVM_REG,
                     alpha=opacity,
                     color='cyan',
                     error_kw=error_config,
                     label='SVM')
    rects6 = plt.bar(index + bar_width, RF_FINAL - RF_REG, bar_width, bottom=RF_REG,
                     alpha=opacity,
                     color='lime',
                     error_kw=error_config,
                     label='RF')
    rects7 = plt.bar(index + 2 * bar_width, ANN_FINAL - ANN_REG, bar_width, bottom=ANN_REG,
                     alpha=opacity,
                     color='yellow',
                     error_kw=error_config,
                     label='ANN')
    rects8 = plt.bar(index + 3 * bar_width, XGB_FINAL - XGB_REG, bar_width, bottom=XGB_REG,
                     alpha=opacity,
                     color='firebrick',
                     error_kw=error_config,
                     label='XGB')

    SVM_FINAL = [0.754, 0.793, 0.390, 0.690, 5.909, 0.647]
    RF_FINAL = [0.760, 0.813, 0.384, 0.701, 6.359, 0.632]
    ANN_FINAL = [0.741, 0.771, 0.403, 0.673, 6.831, 0.591]
    XGB_FINAL = [0.770, 0.806, 0.370, 0.707, 6.773, 0.619]

    for rects, data in zip([rects5, rects6, rects7, rects8], [SVM_FINAL, RF_FINAL, ANN_FINAL, XGB_FINAL]):
        for rect, h in zip(rects, data):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height,
                     str('%.3f' % h), size=fontsize, ha='center', va='bottom')
    # plt.xlim([-0.5, 6.5])
    plt.ylim([0, 1.05])
    # plt.yticks(np.linspace(0.6, 1, 6),  fontsize=axissize)
    plt.xticks(index + bar_width, ('ACC', 'POD', 'FAR', 'F', 'RMSE', 'R'), fontsize=axissize)
    plt.rcParams.update({'font.size': fontsize})
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    Basin_name = 'DongJ'
    BN = 'DJ'
    BS = 'F'
    in_path = '../data'
    processed_path = '../data/processed_' + BN
    DEA_path = processed_path + '/DEA'
    sta_path = '../data/Station_China.xls'
    figure_path = '../figure'
    model_path = '../model'

    Basin_FH = [35, 39.5, 110, 114]
    Basin_DJ = [22.5, 25.5, 113.5, 116]
    SpaceScale = 0.1
    Figindex = [3, 4, 5]
    Si = 59293

    # Evaluate_STA(DEA_path, processed_path, sta_path, processed_path, BN)
    # get_score(processed_path, DEA_path, Basin_FH, SpaceScale, Basin_name, BN, BS)
    # Boxplot(processed_path, figure_path, BN)
    ImportancePlot(model_path, figure_path)
    # for i in Figindex:
    # ScorePlot(in_path, figure_path, Basin_DJ, Basin_FH, SpaceScale, 'F')
    # DistributionPlot(figure_path)
    # TimeseriesPlot(DEA_path, processed_path, sta_path, figure_path, BN, Si)
    # Statisticplot()
    # Statisticplot_reg()


if __name__ == '__main__':
    main()

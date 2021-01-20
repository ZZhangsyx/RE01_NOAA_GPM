# coding=utf-8
"""
Author: Zhang Zhi
Email: zhangzh49@mail2.sysu.edu.cn
Date: 2021/1/4
"""

import numpy as np
import pandas as pd
import os
import ee
import scipy.io as scio
import matplotlib.pyplot as plt
import shapefile
from datetime import datetime
import glob
import warnings
warnings.filterwarnings("ignore")


def isPointinPolygon(point, rangelist):
    """
    Determine whether a point is within the polygon
    :param point: Input point
    :param rangelist: Polygon range list like [[0,0],[1,1],[0,1],[0,0]] [1,0.8]
    :return: Ture/False
    """
    lonlist, latlist = [], []
    for i in range(len(rangelist) - 1):
        lonlist.append(rangelist[i][0])
        latlist.append(rangelist[i][1])
    maxlon, minlon = max(lonlist), min(lonlist)
    maxlat, minlat = max(latlist), min(latlist)

    if (point[0] > maxlon or point[0] < minlon or
            point[1] > maxlat or point[1] < minlat):
        return False
    count = 0
    point1 = rangelist[0]
    for i in range(1, len(rangelist)):
        point2 = rangelist[i]
        if (point[0] == point1[0] and point[1] == point1[1]) or (
                point[0] == point2[0] and point[1] == point2[1]):
            return False
        if (point1[1] < point[1] <= point2[1]) or (
                point1[1] >= point[1] > point2[1]):
            point12lng = point2[0] - (point2[1] - point[1]) * \
                         (point2[0] - point1[0]) / (point2[1] - point1[1])
            if point12lng == point[0]:
                return False
            if point12lng < point[0]:
                count += 1
        point1 = point2
    if count % 2 == 0:
        return False
    else:
        return True


def ShpToBoundary(Basin, Scale, shp_path, save_path, Basin_name, isShow=False):
    """
    :return Create two files: 1)points boundary;
                              2)a m*n matrix determine whether a point is in boundary.
    :param Basin: Range of output matrix [list] like [35, 39.5, 110, 114]
    :param Scale: Spatial resolution [float] like 0.1
    :param shp_path: Input shapefile path
    :param save_path: Output npy file path
    :param Basin_name: Basin name [str]
    :param isShow: Whether show the output matrix
    """
    lat_bs = np.arange(Basin[0] + Scale / 2, Basin[1], Scale)
    lon_bs = np.arange(Basin[2] + Scale / 2, Basin[3], Scale)
    shp = shapefile.Reader(shp_path)
    points = shp.shape().points
    np.save(os.path.join(save_path, Basin_name + '_point.npy'), points)

    isBasin = np.zeros((len(lat_bs), len(lon_bs)))
    for i in range(len(lat_bs)):
        for j in range(len(lon_bs)):
            isBasin[i, j] = isPointinPolygon([lon_bs[j], lat_bs[i]], points)
    np.save(os.path.join(save_path, 'is' + Basin_name + '.npy'), isBasin[::-1, :])

    if isShow:
        plt.imshow(isBasin[::-1, :])
        plt.xticks([])
        plt.yticks([])
        plt.title('BASIN ' + Basin_name, fontsize=14)
        plt.show()


def GEE_download(point_path, Satellite_name, Basin_name, BN, StartDate, EndDate,
                 TimeScale, GEE_Web, bands):
    """
    :return Download GEE data as npy.
    :param point_path: Boundary data path
    :param Satellite_name: GEE data name [str]
    :param Basin_name: Basin name [str]
    :param BN: Basin name's abbreviation [str]
    :param StartDate: Download data start date [str] like "2018-09-22"
    :param EndDate: Download data end date [str] like "2018-09-22"
    :param TimeScale: Time resolution [float] like 0.1
    :param GEE_Web: GEE data's website
    :param bands: Define the bands of GEE data which need to be downloaded
    """
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'
    ee.Initialize()

    out_path = os.path.join(point_path, Satellite_name + '_' + BN)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    point = np.load(os.path.join(point_path, Basin_name + '_point.npy'))
    roi = ee.Geometry.Polygon(point.tolist(), None, False)
    Time_str = [datetime.strftime(x, '%Y-%m-%d')
                for x in list(pd.date_range(start=StartDate, end=EndDate))]

    for itime in Time_str:
        Date_range = ee.Date(itime).getRange(TimeScale)
        Data_day = ee.ImageCollection(GEE_Web).filter(
            ee.Filter.date(Date_range)).select(bands)
        ID = ee.ImageCollection(Data_day).aggregate_array('system:id').getInfo()
        for band in bands:
            value = []
            for id in ID:
                value.append(
                    np.array(
                        ee.Image(id).clip(roi).sampleRectangle(
                            defaultValue=-128).get(band).getInfo()))
            value = np.array(value, dtype=np.float32)
            value[value == -128] = np.nan
            Data_mean = np.nanmean(value, axis=0)
            Data_max = np.nanmax(value, axis=0)

            if not os.path.exists(os.path.join(out_path, band)):
                os.mkdir(os.path.join(out_path, band))
            np.save(os.path.join(out_path, band, band + itime + 'mean.npy'), Data_mean)
            np.save(os.path.join(out_path, band, band + itime + 'max.npy'), Data_max)
            print('\rNow downloading file：' + itime + band, end="")


def FillFile(in_path, strlist):
    """
    :param in_path: Data path
    :param strlist: Name of files [list]
    :return: Complete missing files
    """
    data = np.load(os.path.join(in_path, strlist[0]))
    data0 = np.zeros_like(data)
    for strl in strlist:
        if not os.path.isfile(os.path.join(in_path, strl)):
            np.save(os.path.join(in_path, strl), data0)


def Create_Dataset_Satellite(in_path, save_path, StartDate, EndDate, Satellite_name, bands, BN):
    """
    :return: Fuse the GEE data of each band into a dataset
    :param in_path: GEE download data path
    :param save_path: Data save path
    :param StartDate: Download data start date [str] like "2018-09-22"
    :param EndDate: Download data end date [str] like "2018-09-22"
    :param Satellite_name: GEE data name [str]
    :param bands: Define the bands of GEE data which need to be downloaded
    :param BN: Basin name's abbreviation [str]
    """
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    Datelist = [datetime.strftime(x, '%Y-%m-%d')
                for x in list(pd.date_range(start=StartDate, end=EndDate))]
    for band in bands:
        for por in ['max', 'mean']:
            band_path = os.path.join(in_path, Satellite_name + '_' + BN, band)
            strlist = []
            for date in Datelist:
                strlist.append(band + date + por + '.npy')
            FillFile(band_path, strlist)

            filename = '*' + por + '*.npy'
            files = glob.glob(os.path.join(band_path, filename))

            Bindex = 0
            data_t1 = np.load(files[Bindex], allow_pickle=True)
            while data_t1.size <= 1:
                Bindex = Bindex+1
                data_t1 = np.load(files[Bindex], allow_pickle=True)
            data0 = data_t1[np.newaxis, :]

            for file in files[1:]:
                data_t2 = np.load(file, allow_pickle=True)
                if data_t2.size > 1:
                    data_t1 = data_t2
                data0 = np.concatenate((data0, data_t1[np.newaxis, :]), axis=0)

            np.save(os.path.join(save_path, band + '_' + por + '_set.npy'), data0)
            print('\rCreating dataset：' + band + '_' + por, end="")


def Create_Dataset_CMPA(in_path, out_path, Basin_name, Basin, Scale, CN, Satellite_name='CMPA'):
    """
    :param in_path: CMPA input path
    :param out_path: save path
    :param Satellite_name: [str] 'CMPA'
    :param Basin_name: Basin name [str]
    :param Basin: Range of output matrix [list] like [35, 39.5, 110, 114]
    :param Scale: Space resolution
    :param CN: Range of CMPA dataset's longitude and latitude [list] [15, 59, 70, 140]
    :return: Fuse the CMPA data into a dataset
    """
    if not os.path.isfile(os.path.join(in_path, Satellite_name + '_CN_15-18.npy')):
        listpath = os.path.join(in_path, Satellite_name)
        listname = os.listdir(listpath)
        cmpa = np.zeros((1, 440, 700))
        for i in listname:
            data = np.array(scio.loadmat(os.path.join(listpath, i))['pre_day'])
            data = data[np.newaxis, :]
            cmpa = np.concatenate((cmpa, data))
            print('\rRunning:'+str(i), end="")

        cmpa = cmpa[1:, :, :]
        np.save(os.path.join(in_path, Satellite_name + '_CN_15-18.npy'), cmpa)
    else:
        cmpa = np.load(os.path.join(in_path, Satellite_name + '_CN_15-18.npy'))

    lat_bs = np.arange(Basin[0] + Scale/2, Basin[1], Scale)
    lon_bs = np.arange(Basin[2] + Scale/2, Basin[3], Scale)
    lat_index = int((lat_bs[0] - CN[0] - 0.05)/0.1)
    lon_index = int((lon_bs[0] - CN[2] - 0.05)/0.1)

    CMPA_Basin = cmpa[:, lat_index:lat_index+len(lat_bs), lon_index:lon_index+len(lon_bs)]
    np.save(os.path.join(out_path, Satellite_name + '_' + Basin_name + '.npy'), CMPA_Basin[:, ::-1, :])


def Create_Dataset_GPM(in_path, out_path, StartDate, EndDate, Basin_name, Basin, Scale, CN, Satellite_name='GPM'):
    """
    :param in_path: GPM input path
    :param out_path: save path
    :param StartDate: Download data start date [str] like "2018-09-22"
    :param EndDate: Download data end date [str] like "2018-09-22"
    :param Satellite_name: [str] 'GPM'
    :param Basin_name: Basin name [str]
    :param Basin: Range of output matrix [list] like [35, 39.5, 110, 114]
    :param Scale: Space resolution
    :param CN: Range of GPM dataset's longitude and latitude [list] [15, 59, 70, 140]
    :return: Fuse the GPM data into a dataset
    """
    if not os.path.isfile(os.path.join(in_path, Satellite_name + '_CN_15-18.npy')):
        time_array = pd.date_range(start=datetime(StartDate), end=datetime(EndDate))
        cmpa = np.zeros((1, 440, 700))

        for td in time_array:
            td_str = td.strftime("%Y%m%d")
            days = datetime(int(td_str[0:4]), int(td_str[4:6]), int(td_str[6:])) - datetime(int(td_str[0:4]), 1, 1)
            GPM_input = td_str[0:4] + str('%03d' % (days.days + 1))
            data_GPM = np.array(scio.loadmat(
                os.path.join(in_path, Satellite_name, td_str[0:4], GPM_input + ".mat"))['pre_day'])

            data = data_GPM[np.newaxis, :]
            cmpa = np.concatenate((cmpa, data))
            print('\rRunning:'+td_str, end="")

        cmpa = cmpa[1:, :, :]
        np.save(os.path.join(in_path, Satellite_name + '_CN_15-18.npy'), cmpa)
    else:
        cmpa = np.load(os.path.join(in_path, Satellite_name + '_CN_15-18.npy'))

    lat_bs = np.arange(Basin[0] + Scale/2, Basin[1], Scale)
    lon_bs = np.arange(Basin[2] + Scale/2, Basin[3], Scale)
    lat_index = int((lat_bs[0] - CN[0] - 0.05)/0.1)
    lon_index = int((lon_bs[0] - CN[2] - 0.05)/0.1)

    CMPA_Basin = cmpa[:, lat_index:lat_index+len(lat_bs), lon_index:lon_index+len(lon_bs)]
    np.save(os.path.join(out_path, Satellite_name + '_' + Basin_name + '.npy'), CMPA_Basin[:, ::-1, :])


def Create_Dataset_GPF(Raw_path, out_path, StartDate, EndDate, Basin_name, Basin, Scale, BN, CN, Satellite_name='GPF'):
    """
    :param Raw_path: GPF input path
    :param out_path: save path
    :param StartDate: Download data start date [str] like "2018-09-22"
    :param EndDate: Download data end date [str] like "2018-09-22"
    :param Satellite_name: [str] 'GPF'
    :param Basin_name: Basin name [str]
    :param Basin: Range of output matrix [list] like [35, 39.5, 110, 114]
    :param Scale: Space resolution
    :param BN: Basin name's abbreviation
    :param CN: Range of GPF dataset's longitude and latitude [list] [15, 59, 70, 140]
    :return: Fuse the GPF data into a dataset
    """
    if not os.path.isfile(os.path.join(Raw_path, Satellite_name + '_CN_15-18.npy')):
        time_array = pd.date_range(start=StartDate, end=EndDate)
        cmpa = np.zeros((1, 400, 700))

        for td in time_array:
            td_str = td.strftime("%Y%m%d")
            days = datetime(int(td_str[0:4]), int(td_str[4:6]), int(td_str[6:])) - datetime(int(td_str[0:4]), 1, 1)
            GPM_input = td_str[0:4] + str('%03d' % (days.days + 1))
            data_GPM = np.array(scio.loadmat(
                os.path.join(Raw_path, Satellite_name, td_str[0:4], GPM_input + ".mat"))['prec_select']).T

            data = data_GPM[np.newaxis, :]
            cmpa = np.concatenate((cmpa, data))
            print('\r已完成' + td_str, end="")  # 打印当前进度信息

        cmpa = cmpa[1:, :, :]
        np.save(os.path.join(Raw_path, Satellite_name + '_CN_15-18.npy'), cmpa)
    else:
        cmpa = np.load(os.path.join(Raw_path, Satellite_name + '_CN_15-18.npy'))

    lat_bs = np.arange(Basin[0] + Scale / 2, Basin[1], Scale)
    lon_bs = np.arange(Basin[2] + Scale / 2, Basin[3], Scale)
    lat_index = int((lat_bs[0] - CN[0] - 0.05) / 0.1)
    lon_index = int((lon_bs[0] - CN[2] - 0.05) / 0.1)

    CMPA_Basin = cmpa[:, lat_index:lat_index + len(lat_bs), lon_index:lon_index + len(lon_bs)]
    CMPA_Basin = CMPA_Basin[:, ::-1, :]
    np.save(os.path.join(out_path, Satellite_name + '_' + Basin_name + '.npy'), CMPA_Basin)

    isBasin = np.load(os.path.join('../data', 'is' + Basin_name + '.npy'))
    grid_num = int(isBasin.sum())

    # Time
    Datelist = [datetime.strftime(x, '%Y-%m-%d')
                for x in list(pd.date_range(start=StartDate, end=EndDate))]
    time_num = len(Datelist)
    data = pd.DataFrame(Datelist * grid_num, columns=['Time'])

    # Location
    lat_list, lon_list = [], []
    for ilat in range(len(lat_bs)):
        for jlon in range(len(lon_bs)):
            if isBasin[ilat, jlon]:
                lat_grid = lat_bs[ilat]
                lon_grid = lon_bs[jlon]
                lat_list.append([lat_grid] * time_num)
                lon_list.append([lon_grid] * time_num)
    lat_list, lon_list = np.array(lat_list).reshape(-1), np.array(lon_list).reshape(-1)
    data['Lat'], data['Lon'] = lat_list, lon_list

    band_list = []
    for ilat in range(len(lat_bs)):
        for jlon in range(len(lon_bs)):
            if isBasin[ilat, jlon]:
                band_list.append(CMPA_Basin[:, ilat, jlon])
    band_list = np.array(band_list).reshape(-1)

    data['GPM-F'] = band_list
    data.to_csv(os.path.join('../data/processed_' + BN, Satellite_name + '_' + BN + '.csv'), index=False)


def Create_Dataset_Sta(in_path, out_path, STA_name, Basin_name, StartDate, EndDate, BN):
    """
    :param in_path: STA path
    :param out_path: save path
    :param STA_name: STA table path
    :param Basin_name: Basin name
    :param StartDate: Download data start date [str] like "2018-09-22"
    :param EndDate: Download data end date [str] like "2018-09-22"
    :param BN: Basin name's abbreviation
    :return: Fuse the STA data into a dataset
    """
    station = pd.read_excel(os.path.join(in_path, 'Station_China.xls'), header=2)
    lat_sta = list(station['纬度'].astype(int) / 100)
    lon_sta = list(station['经度'].astype(int) / 100)
    index_sta = list(station['区站号'].astype(int))
    name_sta = list(station['站名'])

    points = np.load(os.path.join('../data', Basin_name + '_point.npy'))
    STA_index = []
    for i in range(len(lat_sta)):
        if isPointinPolygon([lon_sta[i], lat_sta[i]], points):
            STA_index.append(index_sta[i])

    Datelist = [datetime.strftime(x, '%Y-%m-%d')
                for x in list(pd.date_range(start=StartDate, end=EndDate))]
    Sta_csv = pd.DataFrame(Datelist, columns=['Time'])

    for si in STA_index:
        pre_out = []
        for year in range(2015, 2019):
            for month in range(1, 13):
                data = pd.read_table(STA_name + str(year) + str(month).zfill(2) + '.TXT',
                                     header=None, encoding='gb2312', delim_whitespace=True)
                data1 = data.loc[data.iloc[:, 0] == si]
                pre = data1.iloc[:, 9]
                pre = list(pre)
                pre_out.extend(pre)

        pre_out = np.array(pre_out).reshape([-1])
        pre_out[pre_out == 32700] = 1
        pre_out[pre_out == 32766] = 0
        pre_out[pre_out > 32000] = pre_out[pre_out > 32000] - 32000
        pre_out[pre_out > 31000] = pre_out[pre_out > 31000] - 31000
        pre_out[pre_out > 30000] = pre_out[pre_out > 30000] - 30000
        pre_out = pre_out / 10
        Sta_csv[str(si)] = pre_out

    Sta_csv.to_csv(os.path.join(out_path, 'STA_' + BN + '.csv'))


def Create_csv(in_path, Basin, Scale, Basin_name, BN, StartDate, EndDate):
    """
    :param in_path: input data path
    :param Basin: Range of output matrix [list] like [35, 39.5, 110, 114]
    :param Scale: Space resolution
    :param Basin_name: Basin name
    :param StartDate: Download data start date [str] like "2018-09-22"
    :param EndDate: Download data end date [str] like "2018-09-22"
    :param BN: Basin name's abbreviation
    :return: Create csv file contain all the data
    """
    lat_bs = np.arange(Basin[0] + Scale / 2, Basin[1], Scale)
    lon_bs = np.arange(Basin[2] + Scale / 2, Basin[3], Scale)
    isBasin = np.load(os.path.join(in_path, 'is' + Basin_name + '.npy'))
    grid_num = int(isBasin.sum())

    # Time
    Datelist = [datetime.strftime(x, '%Y-%m-%d')
                for x in list(pd.date_range(start=StartDate, end=EndDate))]
    time_num = len(Datelist)
    data = pd.DataFrame(Datelist * grid_num, columns=['Time'])

    # Location
    lat_list, lon_list = [], []
    for ilat in np.arange(len(lat_bs)):
        for jlon in np.arange(len(lon_bs)):
            if isBasin[ilat, jlon]:
                lat_grid = lat_bs[ilat]
                lon_grid = lon_bs[jlon]
                lat_list.append([lat_grid] * time_num)
                lon_list.append([lon_grid] * time_num)
    lat_list, lon_list = np.array(lat_list).reshape(-1), np.array(lon_list).reshape(-1)
    data['Lat'], data['Lon'] = lat_list, lon_list

    listpath = os.path.join(in_path, 'dataset_' + BN)
    listname = os.listdir(listpath)

    for band in listname:
        data_band = np.load(os.path.join(listpath, band))
        band_list = []
        for ilat in range(len(lat_bs)):
            for jlon in range(len(lon_bs)):
                if isBasin[ilat, jlon]:
                    if band == 'CMPA_' + Basin_name + '.npy' or band == 'GPM_' + Basin_name + '.npy':
                        band_list.append(data_band[:, ilat, jlon])
                    else:
                        band_list.append(data_band[:, ilat - 5, jlon - 5])

        band_list = np.array(band_list).reshape(-1)
        data[band[:-8]] = band_list
        print('\r当前进度：' + band, end="")

    data.to_csv(os.path.join(in_path, 'data_' + BN + '.csv'), index=False)


def find_outliers_by_3segama(data, fea):
    """
    :param data: Input data
    :param fea: Feature
    :return: Define whether a value in feature is outliers
    """
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    data[fea+'_outliers'] = data[fea].apply(lambda x: str(
        'Outliers') if x > upper_rule or x < lower_rule else 'Normal value')
    return data


def Csv_DEA(in_path, out_path, Basin_name, BN):
    """
    :param in_path: Csv path
    :param out_path: Save path
    :param Basin_name: Basin name
    :param BN: Basin name's abbreviation
    :return: DEA of csv data
    """
    # load data
    data = pd.read_csv(os.path.join(in_path, 'data_' + BN + '.csv'))
    CMPA = data['CMPA_' + Basin_name]
    GPM = data['GPM_' + Basin_name]
    isRain_CMPA = np.array(data['CMPA_' + Basin_name])
    isRain_CMPA[isRain_CMPA > 0] = 1

    # filling missing data
    data_nan = data.drop(['CMPA_' + Basin_name, 'GPM_' + Basin_name], axis=1)
    data_nan.dropna(axis='columns', thresh=500000, inplace=True)
    data_nan['isRain_CMPA'] = isRain_CMPA
    data_nan['CMPA_' + Basin_name] = CMPA
    data_nan['GPM_' + Basin_name] = GPM
    data_nan.dropna(axis=0, thresh=45, inplace=True)
    numerical_fea = list(data_nan.select_dtypes(exclude=['object']).columns)
    data_nan = data_nan.fillna(data_nan.median())

    # dealing with outliers
    data_cp = data_nan.copy()
    for fea in numerical_fea[:-3]:
        data_cp = find_outliers_by_3segama(data_cp, fea)
        data_cp = data_cp[data_cp[fea + '_outliers'] == 'Normal value']
        data_cp = data_cp.reset_index(drop=True)
        data_cp.drop([fea + '_outliers'], axis=1, inplace=True)

    # delete one-value feature
    one_value_fea = [col for col in data_cp.columns if data_cp[col].nunique() <= 1]
    for fea in one_value_fea:
        data_cp.drop(fea, axis=1, inplace=True)

    # save some feature
    CMPA_cp = data_cp['CMPA_' + Basin_name]
    isRain_CMPA_cp = data_cp['isRain_CMPA']
    data_cp['Time'] = pd.to_datetime(data_cp['Time'])
    data_cp = data_cp.set_index('Time')
    data_cp['CMPA_' + Basin_name].to_csv(os.path.join(out_path, 'CMPA.csv'))
    data_cp['GPM_' + Basin_name].to_csv(os.path.join(out_path, 'GPM.csv'))
    data_cp['Lat'].to_csv(os.path.join(out_path, 'Lat.csv'))
    data_cp['Lon'].to_csv(os.path.join(out_path, 'Lon.csv'))

    # normalization
    data_cp.drop(['CMPA_' + Basin_name, 'isRain_CMPA'], axis=1, inplace=True)
    data_cp = (data_cp - data_cp.min()) / (data_cp.max() - data_cp.min())

    # save DEA csv
    data_cp['isRain_CMPA'] = isRain_CMPA_cp.values
    data_cp['CMPA_' + Basin_name] = CMPA_cp.values
    data_cp.to_csv(os.path.join(out_path, 'data_DEA1.csv'))


#    Main    #
def main():
    # Define variables
    shp_path = r'L:/东江/Fenhe_line.shp'
    save_path = '../data'
    Raw_path = '../data/Raw'
    STA_path = '../data/Raw/2400sta/说明文档'
    STA_name = r'I:\RE01_NOAA_GPM\data\Raw\2400sta\PRE\SURF_CLI_CHN_MUL_DAY-PRE-13011-'

    Basin_name = 'Fenhe'
    BN = 'FH'
    Basin = [35, 39.5, 110, 114]
    CN = [15, 59, 70, 140]

    Satellite_name = 'NOAA'
    GEE_Web = 'NOAA/CDR/PATMOSX/V53'
    bands = [
        'cld_emiss_acha', 'cld_height_acha', 'cld_height_uncer_acha',
        'cld_opd_acha', 'cld_opd_dcomp', 'cld_opd_dcomp_unc',
        'cld_press_acha',
        'cld_reff_acha', 'cld_reff_dcomp', 'cld_reff_dcomp_unc',
        'cld_temp_acha', 'cloud_fraction', 'cloud_fraction_uncertainty',
        'cloud_probability', 'cloud_transmission_0_65um',
        'cloud_type', 'cloud_water_path', 'land_class',
        'refl_0_65um', 'refl_0_65um_stddev_3x3', 'refl_0_86um', 'refl_1_60um',
        'refl_3_75um', 'surface_temperature_retrieved',
        'snow_class', 'surface_type',
        'temp_11_0um', 'temp_11_0um_clear_sky', 'temp_11_0um_stddev_3x3',
        'temp_12_0um', 'temp_3_75um'
    ]

    StartDate = "2018-09-22"
    EndDate = "2018-12-31"
    TimeScale = 'day'
    SpaceScale = 0.1
    isShow = True
    dataset_path = '../data/dataset_' + BN
    processed_path = '../data/processed_' + BN
    DEA_path = '../data/processed_' + BN + '/DEA'

    ShpToBoundary(Basin, SpaceScale, shp_path, save_path, Basin_name, isShow=isShow)
    GEE_download(save_path, Satellite_name, Basin_name, BN, StartDate, EndDate,
                 TimeScale, GEE_Web, bands)
    Create_Dataset_Satellite(save_path, dataset_path, StartDate, EndDate, Satellite_name, bands, BN)
    Create_Dataset_CMPA(Raw_path, dataset_path, Basin_name, Basin, SpaceScale, CN)
    Create_Dataset_GPM(Raw_path, dataset_path, StartDate, EndDate, Basin_name, Basin, SpaceScale, CN)
    Create_Dataset_GPF(Raw_path, dataset_path, StartDate, EndDate, Basin_name, Basin, SpaceScale, BN, CN)
    Create_Dataset_Sta(STA_path, processed_path, STA_name, Basin_name, StartDate, EndDate, BN)
    Create_csv(save_path, Basin, SpaceScale, Basin_name, BN, StartDate, EndDate)
    Csv_DEA(save_path, DEA_path, Basin_name, BN)


if __name__ == '__main__':
    main()

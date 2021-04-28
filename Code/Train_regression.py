# coding=utf-8
"""
Author: Zhang Zhi
Email: zhangzh49@mail2.sysu.edu.cn
Date: 2021/1/5
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
import xgboost as xgb


def Display_Score(data, label, isShow=True):
    """
    :param label: Label
    :param data: Predict value
    :param isShow: Whether to plot
    :return: Evaluation indicators: MSE, RMSE, CC, POD, FAR, accuracy, F
    """
    data = np.reshape(data, [-1, ])
    label = np.reshape(label, [-1, ])
    MSE = mean_squared_error(data, label)
    RMSE = np.sqrt(MSE)
    CC = np.corrcoef(data, label)[0, 1]

    A = label.copy()
    B = data.copy()
    A[A != 0] = 1
    B[B != 0] = 1
    A, B = np.array(A), np.array(B)
    num_A = len(A)
    TP = np.sum(np.multiply(A, B))
    TN = np.sum((A + B) == 0)
    FP = np.sum((A - B) == -1)
    FN = np.sum((A - B) == 1)
    POD = TP / np.sum(A)
    FAR = FP / np.sum(B)
    accuracy = (TP + TN) / num_A
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F = precision * recall * 2 / (precision + recall)

    if isShow:
        print('****** Evaluation Score ******')
        print("MSE： ", '%.3f' % MSE)
        print("RMSE： ", '%.3f' % RMSE)
        print("CC： ", '%.3f' % CC)
        print("POD： ", '%.3f' % POD)
        print("FAR： ", '%.3f' % FAR)
        print("ACC： ", '%.3f' % accuracy)
        print("F：   ", '%.3f' % F, '\n')
    return RMSE, CC, POD, FAR, accuracy, F


class Model(object):

    def __init__(self, model_name, basin_name):
        self.name = model_name
        self.basin = basin_name
        self.path = '../Model/' + self.name + '_R_' + self.basin + '.pkl'

    def load(self):
        if os.path.isfile(self.path):
            self.model = joblib.load(self.path)
            return self.model
        else:
            print('Model didnt exsit!')

    def predict(self, x_train, x_test, y_train, y_test):
        print('Predicting ' + self.name + ':')
        train_class_pre = self.model.predict(x_train)
        train_class_pre[train_class_pre < 0] = 0
        result_train = Display_Score(train_class_pre, y_train)

        test_class_pre = self.model.predict(x_test)
        test_class_pre[test_class_pre < 0] = 0
        result_test = Display_Score(test_class_pre, y_test)

        return train_class_pre, test_class_pre, result_train, result_test


def SVM(x_train, x_test, y_train, y_test, BN):
    Model_SVM = Model(model_name='SVM', basin_name=BN)
    if os.path.isfile(Model_SVM.path):
        model = Model_SVM.load()
    else:
        model = SVR()
        model.fit(x_train, y_train)
        joblib.dump(model, Model_SVM.path)
        model = Model_SVM.load()

    train_class_pre, test_class_pre, result_train, result_test = Model_SVM.predict(
        x_train, x_test, y_train, y_test)
    return train_class_pre, test_class_pre, result_train, result_test


def RF(x_train, x_test, y_train, y_test, BN):
    Model_RF = Model(model_name='RF', basin_name=BN)
    if os.path.isfile(Model_RF.path):
        model = Model_RF.load()
    else:
        model = RandomForestRegressor()
        param_distribs = {'n_estimators': randint(low=1, high=200),
                          'max_features': randint(low=1, high=10),
                          }
        rnd_search = RandomizedSearchCV(
            model,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            random_state=2020)
        rnd_search.fit(x_train, y_train)
        model = rnd_search.best_estimator_
        print('Model RFC Fitted!')
        joblib.dump(model, Model_RF.path)
        model = Model_RF.load()

    train_class_pre, test_class_pre, result_train, result_test = Model_RF.predict(
        x_train, x_test, y_train, y_test)
    return train_class_pre, test_class_pre, result_train, result_test


def ANN(x_train, x_test, y_train, y_test, BN):
    checkpoint_ANN = '../model/ANN_R_' + BN + '.ckpt'
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    if os.path.exists(checkpoint_ANN + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_ANN)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_ANN,
                                                     save_weights_only=True,
                                                     save_best_only=True)
    if not os.path.isfile(checkpoint_ANN):
        model.fit(
            np.array(x_train),
            np.array(y_train),
            batch_size=128,
            epochs=50,
            validation_data=[
                np.array(x_test),
                np.array(y_test)],
            validation_freq=1,
            callbacks=[cp_callback])

    train_class_pre = model.predict(x_train)
    train_class_pre[train_class_pre < 0] = 0
    result_train = Display_Score(train_class_pre, y_train)
    test_class_pre = model.predict(x_test)
    test_class_pre[test_class_pre < 0] = 0
    result_test = Display_Score(test_class_pre, y_test)
    print('ANN deal!')
    return train_class_pre, test_class_pre, result_train, result_test


def XGB(x_train, x_test, y_train, y_test, BN):
    Model_XGB = Model(model_name='XGB', basin_name=BN)
    if os.path.isfile(Model_XGB.path):
        model = Model_XGB.load()
    else:
        model = xgb.XGBRegressor(max_depth=5, n_estimators=1000, colsample_bytree=0.8,
                                 subsample=0.8, nthread=10, learning_rate=0.1, min_child_weight=2)
        model.fit(x_train, y_train)
        print('Model XGBC Fitted!')
        joblib.dump(model, Model_XGB.path)
        model = Model_XGB.load()

    train_class_pre, test_class_pre, result_train, result_test = Model_XGB.predict(
        x_train, x_test, y_train, y_test)
    return train_class_pre, test_class_pre, result_train, result_test


def Train_regression(in_path, out_path, BN, BS, isBalance):
    """
    :param in_path: Csv path
    :param out_path: Save path
    :param BN: Basin name's abbreviation [str]
    :param BS: One word to express basin [str]
    :param isBalance: Whether the data is balance in class distribution
    :return: Predicting class value to csv file
    """
    print('Loading data...')
    csv_name = os.path.join(in_path, 'data_DEA1.csv').replace('\\', '/')
    data_pro = pd.read_csv(csv_name)
    data_pro['Time'] = pd.to_datetime(data_pro['Time'])
    data_pro = data_pro.set_index('Time')

    print('Data processing...')
    Train = data_pro['2015':'2017']
    Test = data_pro['2018']
    x_train = Train.drop(['isRain_CMPA', 'CMPA_' + BS], axis=1)
    x_test = Test.drop(['isRain_CMPA', 'CMPA_' + BS], axis=1)

    CMPA = pd.read_csv(os.path.join(in_path, 'CMPA.csv').replace('\\', '/'))
    CMPA['Time'] = pd.to_datetime(CMPA['Time'])
    CMPA = CMPA.set_index('Time')
    CMPA_Train = CMPA['2015':'2017']
    CMPA_Test = CMPA['2018']
    y_train, y_test = CMPA_Train['CMPA_' + BS], CMPA_Test['CMPA_' + BS]

    if not isBalance:
        print('Execute modeling..')
        Test2 = pd.read_csv(os.path.join(out_path, 'Test_class' + BN + '.csv').replace('\\', '/'))
        df_results = pd.DataFrame()
        nonsen, Test2['ANN_reg'], df_results['ANN_Train_reg'], df_results['ANN_Test_reg'] = ANN(x_train, x_test,
                                                                                                y_train, y_test, BN)
        nonsen, Test2['XGB_reg'], df_results['XGB_Train_reg'], df_results['XGB_Test_reg'] = XGB(x_train, x_test,
                                                                                                y_train, y_test, BN)
        nonsen, Test2['SVM_reg'], df_results['SVM_Train_reg'], df_results['SVM_Test_reg'] = SVM(x_train, x_test,
                                                                                                y_train, y_test, BN)
        nonsen, Test2['RF_reg'], df_results['RF_Train_reg'], df_results['RF_Test_reg'] = RF(x_train, x_test,
                                                                                            y_train, y_test, BN)

        model_name = ['SVM', 'RF', 'ANN', 'XGB']
        for dn in model_name:
            Test2[dn + '_cat'] = Test2[dn + '_reg'] * Test2[dn + '_class']
            df_results[dn + 'Test_cat'] = Display_Score(Test2[dn + '_cat'], y_test)

        data = pd.read_csv(os.path.join(in_path, 'GPM.csv').replace('\\', '/'))
        data['Time'] = pd.to_datetime(data['Time'])
        data = data.set_index('Time')
        data_Train = data['2015':'2017']
        data_Test = data['2018']
        Rain_GPM = np.array(data_Train['GPM_' + BS])
        df_results['GPM_Train'] = Display_Score(Rain_GPM, y_train)
        Rain_GPM = np.array(data_Test['GPM_' + BS])
        df_results['GPM_Test'] = Display_Score(Rain_GPM, y_test)

        Test2.to_csv(os.path.join(out_path, 'Test_reg' + BN + '.csv').replace('\\', '/'), index=True)
        df_results.to_csv(os.path.join(out_path, 'Results_reg' + BN + '.csv').replace('\\', '/'), index=True)
    else:
        print('Execute modeling..')
        Train2 = pd.read_csv(os.path.join(out_path, 'Train_class' + BN + '.csv').replace('\\', '/'))
        Test2 = pd.read_csv(os.path.join(out_path, 'Test_class' + BN + '.csv').replace('\\', '/'))
        df_results = pd.DataFrame()
        Train2['SVM_reg'], Test2['SVM_reg'], df_results['SVM_Train_reg'], df_results['SVM_Test_reg'] = SVM(x_train,
                                                                                                           x_test,
                                                                                                           y_train,
                                                                                                           y_test, BN)
        Train2['RF_reg'], Test2['RF_reg'], df_results['RF_Train_reg'], df_results['RF_Test_reg'] = RF(x_train, x_test,
                                                                                                      y_train, y_test, BN)
        Train2['ANN_reg'], Test2['ANN_reg'], df_results['ANN_Train_reg'], df_results['ANN_Test_reg'] = ANN(x_train,
                                                                                                           x_test,
                                                                                                           y_train,
                                                                                                           y_test,BN)
        Train2['XGB_reg'], Test2['XGB_reg'], df_results['XGB_Train_reg'], df_results['XGB_Test_reg'] = XGB(x_train,
                                                                                                           x_test,
                                                                                                           y_train,
                                                                                                           y_test,BN)

        model_name = ['SVM', 'RF', 'ANN', 'XGB']
        for dn in model_name:
            Train2[dn + '_cat'], Test2[dn + '_cat'] = Train2[dn + '_reg'] * Train2[dn + '_class'], Test2[dn + '_reg'] * \
                                                      Test2[dn + '_class']
            df_results[dn + '_Train_cat'], df_results[dn + 'Test_cat'] = Display_Score(Train2[dn + '_cat'],
                                                                                       y_train), Display_Score(
                Test2[dn + '_cat'], y_test)

        data = pd.read_csv(os.path.join(in_path, 'GPM.csv').replace('\\', '/'))
        data['Time'] = pd.to_datetime(data['Time'])
        data = data.set_index('Time')
        data_Train = data['2015':'2017']
        data_Test = data['2018']
        Rain_GPM = np.array(data_Train['GPM_' + BS])
        df_results['GPM_Train'] = Display_Score(Rain_GPM, y_train)
        Rain_GPM = np.array(data_Test['GPM_' + BS])
        df_results['GPM_Test'] = Display_Score(Rain_GPM, y_test)

        Train2.to_csv(os.path.join(out_path, 'Train_reg' + BN + '.csv').replace('\\', '/'), index=True)
        Test2.to_csv(os.path.join(out_path, 'Test_reg' + BN + '.csv').replace('\\', '/'), index=True)
        df_results.to_csv(os.path.join(out_path, 'Results_reg' + BN + '.csv').replace('\\', '/'), index=True)
    print('Code deal!')


def main():
    BN = 'DJ'
    BS = 'DongJ'
    processed_path = '../data/processed_' + BN +'2'
    DEA_path = processed_path + '/DEA'
    isBalance = False

    Train_regression(DEA_path, processed_path, BN, BS, isBalance)


if __name__ == '__main__':
    main()

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
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import randint
import xgboost as xgb


def calculate_acc(A, B, isShow=True):
    """
    :param A: Label
    :param B: Predict value
    :param isShow: Whether to plot
    :return: Evaluation indicators: POD, FAR, accuracy, F1-score
    """
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
        print("POD： ", '%.3f' % POD)
        print("FAR： ", '%.3f' % FAR)
        print("ACC： ", '%.3f' % accuracy)
        print("F：   ", '%.3f' % F, '\n')

    return POD, FAR, accuracy, F


class Model(object):

    def __init__(self, model_name, basin_name):
        """
        :param model_name: Model name
        :param basin_name: Basin name
        """
        self.name = model_name
        self.basin = basin_name
        self.path = '../Model/' + self.name + '_C_' + self.basin + '.pkl'

    def load(self):
        """
        :return: Load model
        """
        if os.path.isfile(self.path):
            self.model = joblib.load(self.path)
            return self.model
        else:
            print('Model didnt exsit!')

    def predict(self, x_train, x_test, y_train, y_test):
        """
        :param x_train: Input feature in training step
        :param x_test: Input feature in testing step
        :param y_train: Output feature in training step
        :param y_test: Output feature in testing step
        :return: Predicting class value (Rain/No rain)
        """
        print('Predicting ' + self.name + ':')
        train_class_pre = self.model.predict(x_train)
        calculate_acc(train_class_pre, y_train)
        test_class_pre = self.model.predict(x_test)
        calculate_acc(test_class_pre, y_test)
        return train_class_pre, test_class_pre


def SVM(x_train, x_test, y_train, y_test, BN):
    """
    :param BN: Basin name's abbreviation [str]
    :param x_train: Input feature in training step
    :param x_test: Input feature in testing step
    :param y_train: Output feature in training step
    :param y_test: Output feature in testing step
    """
    Model_SVM = Model(model_name='SVM', basin_name=BN)
    if os.path.isfile(Model_SVM.path):
        model = Model_SVM.load()
    else:
        model = SVC(C=1, kernel='rbf', shrinking=True, max_iter=-1)
        model.fit(x_train, y_train)
        joblib.dump(model, Model_SVM.path)
        model = Model_SVM.load()

    train_class_pre, test_class_pre = Model_SVM.predict(
        x_train, x_test, y_train, y_test)
    return train_class_pre, test_class_pre


def RF(x_train, x_test, y_train, y_test, BN):
    Model_RF = Model(model_name='RF', basin_name=BN)
    if os.path.isfile(Model_RF.path):
        model = Model_RF.load()
    else:
        model = RandomForestClassifier(random_state=2020)
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

    train_class_pre, test_class_pre = Model_RF.predict(
        x_train, x_test, y_train, y_test)
    return train_class_pre, test_class_pre


def ANN(x_train, x_test, y_train, y_test, BN):
    checkpoint_ANN = '../model/ANN_C_' + BN + '.ckpt'
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False),
        metrics=['sparse_categorical_accuracy'])
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
            epochs=100,
            validation_data=[
                np.array(x_test),
                np.array(y_test)],
            validation_freq=1,
            callbacks=[cp_callback])

    print('Predicting ANN:')
    train_class_pre = model.predict(x_train).argmax(axis=1)
    calculate_acc(train_class_pre, y_train)
    test_class_pre = model.predict(x_test).argmax(axis=1)
    calculate_acc(test_class_pre, y_test)
    return train_class_pre, test_class_pre


def XGB(x_train, x_test, y_train, y_test, BN):
    Model_XGB = Model(model_name='XGB', basin_name=BN)
    if os.path.isfile(Model_XGB.path):
        model = Model_XGB.load()
    else:
        model = xgb.XGBClassifier(
            max_depth=5,
            n_estimators=1000,
            colsample_bytree=0.8,
            subsample=0.8,
            nthread=10,
            learning_rate=0.1,
            min_child_weight=2)
        model.fit(x_train, y_train)
        print('Model XGBC Fitted!')
        joblib.dump(model, Model_XGB.path)
        model = Model_XGB.load()

    train_class_pre, test_class_pre = Model_XGB.predict(
        x_train, x_test, y_train, y_test)
    return train_class_pre, test_class_pre


def Train_class(in_path, out_path, BN, BS, isBalance):
    """
    :param in_path: Csv path
    :param out_path: Save path
    :param BN: Basin name's abbreviation [str]
    :param BS: One word to express basin [str]
    :param isBalance: Whether the data is balance in class distribution
    :return: Predicting class value to csv file
    """
    print('Loading data...')
    csv_name = os.path.join(in_path, 'data_DEA1.csv')
    data_pro = pd.read_csv(csv_name)
    data_pro['Time'] = pd.to_datetime(data_pro['Time'])
    data_pro = data_pro.set_index('Time')

    print('Data processing...')
    Train = data_pro['2015':'2017']
    Test = data_pro['2018']
    y_train, y_test = Train['isRain_CMPA'], Test['isRain_CMPA']
    x_train = Train.drop(['isRain_CMPA', 'CMPA_' + BS], axis=1)
    x_test = Test.drop(['isRain_CMPA', 'CMPA_' + BS], axis=1)

    if not isBalance:
        smo = SMOTE(random_state=42)
        x_train, y_train = smo.fit_sample(x_train, y_train)

        print('Execute modeling..')
        nonsen, Test['SVM_class'] = SVM(x_train, x_test, y_train, y_test, BN)
        nonsen, Test['RF_class'] = RF(x_train, x_test, y_train, y_test, BN)
        nonsen, Test['ANN_class'] = ANN(x_train, x_test, y_train, y_test, BN)
        nonsen, Test['XGB_class'] = XGB(x_train, x_test, y_train, y_test, BN)

        # Compare with GPM
        isRain_GPM = np.array(Test['GPM_' + BS])
        isRain_GPM[isRain_GPM > 0] = 1
        Test['GPM_class'] = isRain_GPM
        calculate_acc(isRain_GPM, y_test)

        # Save
        Test.to_csv(os.path.join(out_path, 'Test_class' + BN + '.csv'), index=True)

    else:
        print('Execute modeling..')
        Train['SVM_class'], Test['SVM_class'] = SVM(x_train, x_test, y_train, y_test)
        Train['RF_class'], Test['RF_class'] = RF(x_train, x_test, y_train, y_test)
        Train['ANN_class'], Test['ANN_class'] = ANN(x_train, x_test, y_train, y_test)
        Train['XGB_class'], Test['XGB_class'] = XGB(x_train, x_test, y_train, y_test)

        # Compare with GPM
        isRain_GPM = np.array(Train['GPM_' + BS])
        isRain_GPM[isRain_GPM > 0] = 1
        Train['GPM_class'] = isRain_GPM
        calculate_acc(isRain_GPM, y_train)

        isRain_GPM = np.array(Test['GPM_' + BS])
        isRain_GPM[isRain_GPM > 0] = 1
        Test['GPM_class'] = isRain_GPM
        calculate_acc(isRain_GPM, y_test)

        # Save
        Train.to_csv(os.path.join(out_path, 'Train_class' + BN + '.csv'), index=True)
        Test.to_csv(os.path.join(out_path, 'Test_class' + BN + '.csv'), index=True)

    print('Code deal!')


def main():
    BN = 'FH'
    BS = 'F'
    processed_path = '../data/processed_' + BN
    DEA_path = processed_path + '/DEA'
    isBalance = False

    Train_class(DEA_path, processed_path, BN, BS, isBalance)


if __name__ == '__main__':
    main()

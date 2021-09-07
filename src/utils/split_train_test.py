# -*- coding: utf-8 -*-
from os import path

import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedShuffleSplit

import src.utils.input_output as io
from src.utils.logger import Logger
from src.utils.paths import PATH_DATA_PROCESSED


def split_train_test(X, y, test_set_percent):
    """
    Stratified randomized folds preserving the % of samples for each class
    :param X: independence data
    :param y: dependence data
    :return: independence/ dependence data split by train and test (20%)
    """

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_set_percent, random_state=9)
    for train_idx, test_idx in split.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    return X_train, y_train, X_test, y_test


def SMOTE_under_sampling(X, y, oversample_percent, undersample_percent):
    """
    Oversample with SMOTE and random undersampling for imbalanced dataset
    :param X:
    :param y:
    :return:
    """

    over = SMOTE(sampling_strategy=oversample_percent)
    under = RandomUnderSampler(sampling_strategy=undersample_percent)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X_smote, y_smote = pipeline.fit_resample(X, y)
    return X_smote, y_smote


def generate_data_for_model_training(data, test_set_percent, oversample_percent, undersample_percent):
    Logger.info('Start generating training data')

    X = data.drop(['Time', 'Class'], axis=1)
    y = data['Class']

    # standardize 'Amount' column
    min_amount = tf.reduce_min(X['Amount'])
    max_amount = tf.reduce_max(X['Amount'])
    X['Amount_norm'] = (X['Amount'] - min_amount) / (max_amount - min_amount)
    X = X.drop(['Amount'], axis=1)

    # split train & test
    X_train, y_train, X_test, y_test = split_train_test(X, y, test_set_percent)

    # balance training set 0/1 ratio
    # X_smote, y_smote = SMOTE_under_sampling(X_train, y_train, oversample_percent, undersample_percent)

    io.write_pickle(X_train, path.join(PATH_DATA_PROCESSED, 'X_train.pkl'))
    io.write_pickle(y_train, path.join(PATH_DATA_PROCESSED, 'y_train.pkl'))
    io.write_pickle(X_test, path.join(PATH_DATA_PROCESSED, 'X_test.pkl'))
    io.write_pickle(y_test, path.join(PATH_DATA_PROCESSED, 'y_test.pkl'))


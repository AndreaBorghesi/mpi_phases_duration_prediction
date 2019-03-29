'''
Predict the duration of MPI calls
Using ML

Andrea Borghesi, andrea.borghesi3@unibo.it
    University of Bologna
    2019-02-05
'''
#!/usr/bin/python

import numpy
import sys
import os
import subprocess
import ast
import matplotlib 
import matplotlib.pyplot as plt
from pyDOE import *
import pickle
from matplotlib.colors import LogNorm
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import explained_variance_score, accuracy_score
from sklearn import linear_model
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import LogisticRegression,SGDRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import time
import my_util

main_dir = '/home/b0rgh/collaborations_potential/dcesarini/'

'''
Train ML model in order to predict MPI call duration
'''
def train_ML_MPI_durs(df_orig, verbose=True):
    df = df_orig.copy()
    target = df['time_duration']
    #del df['inst_ret']
    del df['time_duration']

    train_data, test_data, train_target, test_target = train_test_split(
            df, target, test_size = 0.3, random_state = 42)

    #regr = RandomForestRegressor(n_estimators=100)
    regr = RandomForestRegressor()

    regr.fit(train_data,train_target)

    predicted = regr.predict(test_data)
    actual = test_target.values

    stats_res = my_util.evaluate_predictions(predicted, actual)

    stats_res['feat_importances'] = regr.feature_importances_

    if verbose:
        print("MAE: {0:.3f}".format(stats_res["MAE"]))
        print("MSE: {0:.3f}".format(stats_res["MSE"]))
        print("RMSE: {0:.3f}".format(stats_res["RMSE"]))
        print("MAPE: {0:.3f}".format(stats_res["MAPE"]))
        print("SMAPE: {0:.3f}".format(stats_res["SMAPE"]))
        print("R-squared: {0:.3f}".format(stats_res["R2"]))
        print("MedAE: {0:.3f}".format(stats_res["MedAE"]))
        print("Explained Variance: {0:.3f}".format(stats_res["EV"]))
        print("Accuracy: {0:.2f}".format(stats_res["accuracy"]))
        print("Features importance {}".format(regr.feature_importances_))
        print("Features importance: [")
        for i in range(len(regr.feature_importances_)):
            print("\t{0} -> {1:.3f}".format(list(df)[i],
                regr.feature_importances_[i]))
        print("]")
    return stats_res

'''
Predict the MPI call duration for a single, specific application
Requires a data set of different runs for the same application
'''
def predict_duration_single_application(fname, verbose=True):
    data_dir = main_dir
    filename = data_dir + fname
    df = pd.read_csv(fname, sep=';', header=0)
    del df['inst_ret']

    print(df['']

    stats_res = {}
    n_iter = 20
    for ni in range(n_iter):
        print("Iteration {}".format(ni))

        sr = train_ML_MPI_durs(df, False)

        if len(stats_res) == 0:
            stats_res["MAE"] = sr["MAE"]
            stats_res["MSE"] = sr["MSE"]
            stats_res["RMSE"] = sr["RMSE"]
            stats_res["MAPE"] = sr["MAPE"]
            stats_res["SMAPE"] = sr["SMAPE"]
            stats_res["accuracy"] = sr["accuracy"]
            stats_res["R2"] = sr["R2"]
            stats_res["feat_importances"] = sr["feat_importances"]
        else:
            for k in stats_res.keys():
                if k != 'feat_importances':
                    stats_res[k] += sr[k]
                else:
                    for fii in range(len(stats_res[k])):
                        stats_res[k][fii] += sr[k][fii]
    for k in stats_res.keys():
        if k != 'feat_importances':
            stats_res[k] /= n_iter
        else:
            for fii in range(len(stats_res[k])):
                stats_res[k][fii] /= n_iter

    print(df)

    if verbose:
        print("MAE: {0:.3f}".format(stats_res["MAE"]))
        print("MSE: {0:.3f}".format(stats_res["MSE"]))
        print("RMSE: {0:.3f}".format(stats_res["RMSE"]))
        print("MAPE: {0:.3f}".format(stats_res["MAPE"]))
        print("SMAPE: {0:.3f}".format(stats_res["SMAPE"]))
        print("R-squared: {0:.3f}".format(stats_res["R2"]))
        print("Accuracy: {0:.2f}".format(stats_res["accuracy"]))
        print("Features importance: [")
        for i in range(len(stats_res['feat_importances'])):
            print("\t{0} -> {1:.3f}".format(list(df)[i],
                stats_res['feat_importances'][i]))
        print("]")


'''
Predict the MPI call durations with a data set containing previous runs for
multiple applications
'''
def predict_duration_multi_application(dname, verbose=True):
    data_dir = main_dir + dname
    directory = os.fsencode(data_dir)

    df_list = []

    for f in os.listdir(directory):
        filename = os.fsdecode(f)

        if filename.startswith('mpi_type'):
            continue

        df = pd.read_csv(data_dir + filename, sep=';', header=0)
        df['app_name'] = filename[:-4]

        df_list.append(df)

    all_app_df = pd.concat(df_list)
    all_app_df = my_util.encode_category(all_app_df, 'app_name')
    stats_res = {}
    n_iter = 20
    for ni in range(n_iter):
        print("Iteration {}".format(ni))

        sr = train_ML_MPI_durs(all_app_df, False)

        if len(stats_res) == 0:
            stats_res["MAE"] = sr["MAE"]
            stats_res["MSE"] = sr["MSE"]
            stats_res["RMSE"] = sr["RMSE"]
            stats_res["MAPE"] = sr["MAPE"]
            stats_res["SMAPE"] = sr["SMAPE"]
            stats_res["accuracy"] = sr["accuracy"]
            stats_res["R2"] = sr["R2"]
            stats_res["feat_importances"] = sr["feat_importances"]
        else:
            for k in stats_res.keys():
                if k != 'feat_importances':
                    stats_res[k] += sr[k]
                else:
                    for fii in range(len(stats_res[k])):
                        stats_res[k][fii] += sr[k][fii]
    for k in stats_res.keys():
        if k != 'feat_importances':
            stats_res[k] /= n_iter
        else:
            for fii in range(len(stats_res[k])):
                stats_res[k][fii] /= n_iter

    if verbose:
        print("MAE: {0:.3f}".format(stats_res["MAE"]))
        print("MSE: {0:.3f}".format(stats_res["MSE"]))
        print("RMSE: {0:.3f}".format(stats_res["RMSE"]))
        print("MAPE: {0:.3f}".format(stats_res["MAPE"]))
        print("SMAPE: {0:.3f}".format(stats_res["SMAPE"]))
        print("R-squared: {0:.3f}".format(stats_res["R2"]))
        print("Accuracy: {0:.2f}".format(stats_res["accuracy"]))
        print("Features importance: [")
        for i in range(len(stats_res['feat_importances'])):
            print("\t{0} -> {1:.3f}".format(list(all_app_df)[i],
                stats_res['feat_importances'][i]))
        print("]")

    

def main(argv):
    pred_type = int(argv[0])
    fname = argv[1]

    if pred_type == 0:
        predict_duration_single_application(fname)
    elif pred_type == 1:
        predict_duration_multi_application(fname)

if __name__ == '__main__':
    main(sys.argv[1:])

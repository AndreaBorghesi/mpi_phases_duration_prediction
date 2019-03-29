'''
Predict the duration of MPI calls
Using ML
- new version
- predict the duration of different MPI phases 
    - application
    - synchronization
    - data movement 

Andrea Borghesi
    University of Bologna
    2019-03-29
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
def train_ML_MPI_durs(df_orig, target, verbose=True):
    df = df_orig.copy()
    df[target] *= 100000
    #mask = df[target] < 1
    #df.loc[mask, target] = 1
    #target = np.log(df[target])
    target = df[target]
    del df['time_app']
    del df['time_slack']
    del df['time_mpi']

    train_data, test_data, train_target, test_target = train_test_split(
            df, target, test_size = 0.3, random_state = 42)

    #regr = RandomForestRegressor(n_estimators=100)
    regr = RandomForestRegressor()

    regr.fit(train_data,train_target)

    #predicted = np.exp(regr.predict(test_data))
    #actual = np.exp(test_target.values)
    predicted = regr.predict(test_data)
    actual = test_target.values

    #for i in range(20):
    #    print("{} - {}".format(actual[i], predicted[i]))

    stats_res = my_util.evaluate_predictions(predicted, actual)

    stats_res['feat_importances'] = regr.feature_importances_

    if verbose:
        print(" - MAE: {0:.6f}".format(stats_res["MAE"]))
        print(" - MSE: {0:.6f}".format(stats_res["MSE"]))
        print(" - RMSE: {0:.6f}".format(stats_res["RMSE"]))
        print(" - (Actual mean {0:.6f}, std {1:.3f})".format(np.mean(actual),
            np.std(actual)))
        print(" - MAPE: {0:.3f}".format(stats_res["MAPE"]))
        print(" - SMAPE: {0:.3f}".format(stats_res["SMAPE"]))
        print(" - R-squared: {0:.3f}".format(stats_res["R2"]))
        print(" - MedAE: {0:.3f}".format(stats_res["MedAE"]))
        print(" - Explained Variance: {0:.3f}".format(stats_res["EV"]))
        #print(" - Accuracy: {0:.2f}".format(stats_res["accuracy"]))
        #print(" - Features importance {}".format(regr.feature_importances_))
        print(" - Features importance: [")
        for i in range(len(regr.feature_importances_)):
            print("\t{0} -> {1:.3f}".format(list(df)[i],
                regr.feature_importances_[i]))
        print(" - ]")

    return stats_res

'''
Predict the MPI call duration for a single, specific application
Requires a data set of different runs for the same application
'''
def predict_duration_single_application(fname, verbose=True):
    data_dir = main_dir
    filename = data_dir + fname
    df_full = pd.read_csv(fname, sep=';', header=0)
    size = 1000000
    #df = df_full.sample(size)
    df = df_full
    df = df[(df['time_app'] >= 0.0005)]
    if len(df) > size:
        df = df.sample(size)
    del df['eam_slack           ']
    del df['inst_ret_app']
    del df['inst_ret_slack']
    del df['inst_ret_mpi']

    print("Data frame loaded")

    print("\nPredict time app")
    stat_res_app = train_ML_MPI_durs(df, 'time_app', True)

    print('==============================================================')
    print("\nPredict time slack")
    stat_res_app = train_ML_MPI_durs(df, 'time_slack', True)
    print('==============================================================')
    print("\nPredict time mpi")
    stat_res_app = train_ML_MPI_durs(df, 'time_mpi', True)
    print('==============================================================')
    #stats_res = {}
    #n_iter = 20
    #for ni in range(n_iter):
    #    print("Iteration {}".format(ni))

    #    sr = train_ML_MPI_durs(df, False)

    #    if len(stats_res) == 0:
    #        stats_res["MAE"] = sr["MAE"]
    #        stats_res["MSE"] = sr["MSE"]
    #        stats_res["RMSE"] = sr["RMSE"]
    #        stats_res["MAPE"] = sr["MAPE"]
    #        stats_res["SMAPE"] = sr["SMAPE"]
    #        stats_res["accuracy"] = sr["accuracy"]
    #        stats_res["R2"] = sr["R2"]
    #        stats_res["feat_importances"] = sr["feat_importances"]
    #    else:
    #        for k in stats_res.keys():
    #            if k != 'feat_importances':
    #                stats_res[k] += sr[k]
    #            else:
    #                for fii in range(len(stats_res[k])):
    #                    stats_res[k][fii] += sr[k][fii]
    #for k in stats_res.keys():
    #    if k != 'feat_importances':
    #        stats_res[k] /= n_iter
    #    else:
    #        for fii in range(len(stats_res[k])):
    #            stats_res[k][fii] /= n_iter

    #if verbose:
    #    print("MAE: {0:.3f}".format(stats_res["MAE"]))
    #    print("MSE: {0:.3f}".format(stats_res["MSE"]))
    #    print("RMSE: {0:.3f}".format(stats_res["RMSE"]))
    #    print("MAPE: {0:.3f}".format(stats_res["MAPE"]))
    #    print("SMAPE: {0:.3f}".format(stats_res["SMAPE"]))
    #    print("R-squared: {0:.3f}".format(stats_res["R2"]))
    #    print("Accuracy: {0:.2f}".format(stats_res["accuracy"]))
    #    print("Features importance: [")
    #    for i in range(len(stats_res['feat_importances'])):
    #        print("\t{0} -> {1:.3f}".format(list(df)[i],
    #            stats_res['feat_importances'][i]))
    #    print("]")

def main(argv):
    pred_type = int(argv[0])
    fname = argv[1]

    print(fname)

    if pred_type == 0:
        predict_duration_single_application(fname)
    elif pred_type == 1:
        predict_duration_multi_application(fname)

if __name__ == '__main__':
    main(sys.argv[1:])

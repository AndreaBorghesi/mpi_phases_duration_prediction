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
import csv

main_dir = '/home/b0rgh/collaborations_potential/dcesarini/'
stats_res_dir = main_dir + 'stat_res_dir/'

_bad_pred_threshold = 1000

#_target_type = 'normalTe_normalTr'
#_target_type = 'logTe_logTr'
_target_type = 'normalTe_logTr'

stats_res_dir += _target_type + '/'

'''
Train ML model in order to predict MPI call duration
'''
def train_ML_MPI_durs(df_orig, target, verbose=True):
    df = df_orig.copy()
    df[target] *= 100000
    if _target_type == 'logTe_logTr' or _target_type == 'normalTe_logTr':
        mask = df[target] < 1
        df.loc[mask, target] = 1
        target = np.log(df[target])
    elif _target_type == 'normalTe_normalTr':
        target = df[target]
        
    del df['time_app']
    del df['time_slack']
    del df['time_mpi']

    train_data, test_data, train_target, test_target = train_test_split(
            df, target, test_size = 0.3, random_state = 42)

    #regr = RandomForestRegressor(n_estimators=100)
    regr = RandomForestRegressor()

    regr.fit(train_data,train_target)

    if _target_type == 'normalTe_logTr':
        predicted = np.exp(regr.predict(test_data))
        actual = np.exp(test_target.values)
    else:
        predicted = regr.predict(test_data)
        actual = test_target.values

    #for i in range(20):
    #    if actual[i] != 0:
    #        errp = (abs(predicted[i]-actual[i]))* 100 / abs(actual[i])
    #    else:
    #        errp = 0
    #    print("{} - {} ({})".format(actual[i], predicted[i], errp))

    stats_res = my_util.evaluate_predictions(predicted, actual,
            _bad_pred_threshold)

    stats_res['feat_importances'] = regr.feature_importances_

    if verbose:
        print(" - MAE: {0:.6f}".format(stats_res["MAE"]))
        print(" - MSE: {0:.6f}".format(stats_res["MSE"]))
        print(" - RMSE: {0:.6f}".format(stats_res["RMSE"]))
        print(" - (Actual mean {0:.6f}, std {1:.3f})".format(np.mean(actual),
            np.std(actual)))
        print(" - MAPE: {0:.3f}".format(stats_res["MAPE"]))
        print(" - MAPE no bad preds: {0:.3f}".format(stats_res["MAPE_2"]))
        print(" - SMAPE: {0:.3f}".format(stats_res["SMAPE"]))
        print(" - SMAPE no bad preds: {0:.3f}".format(stats_res["SMAPE_2"]))
        print(" - # bad preds (threshold = {0}): {1} (tot {2});{3:.2f}%".format(
            _bad_pred_threshold, stats_res["nbad_preds"], len(predicted),
            stats_res["nbad_preds"]/len(predicted)*100))
        print(" - (Actual badly predicted mean {0:.6f}, std {1:.3f})".format(
            np.mean(stats_res["badly_predicted"]), 
            np.std(stats_res["badly_predicted"])))
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
    stat_res_app_file = stats_res_dir + fname.split('/')[1].split('.')[0
            ] + '_app.pickle'
    with open(stat_res_app_file, 'wb') as handle:
        pickle.dump(stat_res_app, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('==============================================================')

    print("\nPredict time slack")
    stat_res_slack = train_ML_MPI_durs(df, 'time_slack', True)
    stat_res_slack_file = stats_res_dir + fname.split('/')[1].split('.')[0
            ] + '_slack.pickle'
    with open(stat_res_slack_file, 'wb') as handle:
        pickle.dump(stat_res_slack, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('==============================================================')

    print("\nPredict time mpi")
    stat_res_mpi = train_ML_MPI_durs(df, 'time_mpi', True)
    stat_res_mpi_file = stats_res_dir + fname.split('/')[1].split('.')[0
            ] + '_mpi.pickle'
    with open(stat_res_mpi_file, 'wb') as handle:
        pickle.dump(stat_res_mpi, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('==============================================================')

'''
Compute values relative to the previous MPI phase of the same rank
- iterates over the input csv file and creates a new csv file
'''
def create_new_csv_with_prev(fname):
    wrtxt = 'rank_id;mpi_type;data_in;data_out;num_procs;locality;task_id;'
    wrtxt += 'eam_slack           ;time_app;time_slack;time_mpi;inst_ret_app;'
    wrtxt += 'inst_ret_slack;inst_ret_mpi;prev_time_app;prev_time_slack;'
    wrtxt += 'prev_time_mpi\n'

    prev_rank_id = -1
    prev_task_id = -1
    prev_time_app = -1
    prev_time_slack = -1
    prev_time_mpi = -1

    outfile = 'benchmark_withPrev/' + fname.split('/')[1]
    with open(outfile, 'w+') as of:
        of.write(wrtxt)
    wrtxt = ''

    with open(fname, 'r') as csvfile:
        lines = csvfile.readlines()
        for l in lines[1:]:
            ll = l.split(';')
            rank_id = ll[0]
            task_id = ll[6]
            time_app = ll[8]
            time_slack = ll[9]
            time_mpi = ll[10]

            wrtxt += l.rstrip()

            if rank_id != prev_rank_id:
                prev_rank_id = rank_id
                prev_task_id = task_id
                prev_time_app = -1
                prev_time_slack = -1
                prev_time_mpi = -1
            wrtxt += ';{};{};{}'.format(prev_time_app, prev_time_slack,
                    prev_time_mpi)
            prev_rank_id = rank_id
            prev_task_id = task_id
            prev_time_app = time_app
            prev_time_slack = time_slack
            prev_time_mpi = time_mpi

            wrtxt += '\n'

        with open(outfile, 'a') as of:
            of.write(wrtxt)

#### NOT USED ANYMORE -- I've directly changed the csv files
#'''
#Predict the MPI call duration for a single, specific application
#Requires a data set of different runs for the same application
#- using also information about the duration at the previous iteration
#'''
#def predict_duration_single_application_withPrev(fname, verbose=True):
#    data_dir = main_dir
#    filename = data_dir + fname
#    df_full = pd.read_csv(fname, sep=';', header=0)
#    prev_df_full = compute_prev(df_full)
#    size = 1000000
#    df = prev_df_full
#    df = df[(df['time_app'] >= 0.0005)]
#    if len(df) > size:
#        df = df.sample(size)
#    del df['eam_slack           ']
#    del df['inst_ret_app']
#    del df['inst_ret_slack']
#    del df['inst_ret_mpi']
#
#    print("Data frame loaded")
#
#    print("\nPredict time app")
#    stat_res_app = train_ML_MPI_durs(df, 'time_app', True)
#
#    print('==============================================================')
#    print("\nPredict time slack")
#    stat_res_app = train_ML_MPI_durs(df, 'time_slack', True)
#    print('==============================================================')
#    print("\nPredict time mpi")
#    stat_res_app = train_ML_MPI_durs(df, 'time_mpi', True)
#    print('==============================================================')


def main(argv):
    pred_type = int(argv[0])
    fname = argv[1]

    print(fname)

    if pred_type == 0:
        predict_duration_single_application(fname)
    elif pred_type == 1:
        create_new_csv_with_prev(fname)

if __name__ == '__main__':
    main(sys.argv[1:])

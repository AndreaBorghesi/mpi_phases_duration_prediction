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
#stats_res_dir = main_dir + 'stat_res_dir/using_pred/'
stats_res_dir = main_dir + 'stat_res_dir/no_pred/'
plot_dir = main_dir + 'plots/'

_bad_pred_threshold = 1000

#_target_type = 'normalTe_normalTr'
#_target_type = 'logTe_logTr'
_target_type = 'normalTe_logTr'

stats_res_dir += _target_type + '/'

feature_importances_app = {
        'Rank id': [0.059, 0.009, 0.065, 0.004, 0.021, 0.028, 0.832, 0.002,
            0.132, 0.017],
        'MPI Type': [0.002, 0.194, 0.001, 0.398, 0.001, 0.096, 0, 0.701, 0.018,
            0.008],
        'data in': [0.031, 0.182, 0, 0.2, 0.511, 0.021, 0.007, 0.023, 0.112, 0],
        'data out': [0.516, 0.179, 0.118, 0.397, 0.078, 0.532, 0.033, 0.062,
            0.045, 0.403],
        '# procs': [0, 0, 0, 0, 0, 0, 0, 0, 0.022, 0],
        'Locality': [0, 0, 0, 0, 0, 0, 0, 0, 0.005, 0],
        'Task id': [0.043, 0.001, 0.038, 0.001, 0.001, 0.007, 0.009, 0.001,
            0.185, 0.005],
        'Prev. time app.': [0.135, 0.143, 0.098, 0, 0.329, 0.102, 0.101, 0.019,
            0.235, 0.163],
        'Prev. time slack': [0.108, 0.127, 0.041, 0, 0.03, 0.073, 0.012, 0.119,
            0.095, 0.019],
        'Prev. time MPI': [0.105, 0.166, 0.638, 0, 0.028, 0.141, 0.006, 0.072,
            0.15, 0.385]
        }

feature_importances_slack = {
        'Rank id': [0.013, 0.033, 0.193, 0.027, 0.007, 0.033, 0.141, 0.009,
            0.137, 0.19],
        'MPI Type': [0.001, 0.362, 0.007, 0.472, 0.022, 0.737, 0.275, 0.142,
            0.106, 0.116],
        'data in': [0.001, 0.464, 0, 0.157, 0.052, 0.001, 0.412, 0.842, 0.008,
            0.073],
        'data out': [0.051, 0.362, 0.063, 0.342, 0.794, 0.169, 0, 0, 0.003,
            0.187],
        '# procs': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Locality': [0, 0, 0.002, 0, 0.001, 0, 0, 0, 0, 0],
        'Task id': [0.006, 0.005, 0.121, 0.003, 0.001, 0.003, 0.006, 0.004,
            0.157, 0.08],
        'Prev. time app.': [0.018, 0.007, 0.254, 0, 0.037, 0.037, 0.04, 0.001,
            0.254, .166],
        'Prev. time slack': [0.893, 0.001, .125, 0, .033, .012, .075, .001,
            .124, .119],
        'Prev. time MPI': [0.014, 0.002, .235, 0, .052, .007, .05, .002, .22,
            .119]
        }

feature_importances_mpi = {
        'Rank id': [0.162, 0.013, .047, .15, 0, .001, .013, .016, .145, .092],
        'MPI Type': [0, 0.276, 0, .579, .992, .259, 0, .351, .002, .001],
        'data in': [0.001, 0.453, 0, .026, .001, .005, .0, .15, .027, 0],
        'data out': [0.104, 0.035, .859, .019, .001, .463, .953, .069, .001,
            .816],
        '# procs': [0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0],
        'Locality': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Task id': [0.088, 0.007, .02, .226, 0, .001, .002, .004, .159, .015],
        'Prev. time app.': [0.156, 0.083, .033, 0, .002, .153, .019, .023,
            .214, .043],
        'Prev. time slack': [0.327, 0.095, .009, 0, .003, .103, .002, .381,
            .135, .011],
        'Prev. time MPI': [0.161, 0.038, .032, 0, .001, .007, 0.011, .006, .315,
            .22]
        }

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

'''
Compute values relative to the previous MPI phase of the same rank
- iterates over the input csv file and creates a new csv file
- consider only previous phases with same MPI type
'''
def create_new_csv_with_prev_checkMPItype(fname):
    wrtxt = 'rank_id;mpi_type;data_in;data_out;num_procs;locality;task_id;'
    wrtxt += 'eam_slack           ;time_app;time_slack;time_mpi;inst_ret_app;'
    wrtxt += 'inst_ret_slack;inst_ret_mpi;prev_time_app;prev_time_slack;'
    wrtxt += 'prev_time_mpi\n'

    prev_rank_id = {}
    prev_task_id = {}
    prev_time_app = {}
    prev_time_slack = {}
    prev_time_mpi = {}

    outfile = 'benchmark_withPrev/' + fname.split('/')[1]
    with open(outfile, 'w+') as of:
        of.write(wrtxt)
    wrtxt = ''

    with open(fname, 'r') as csvfile:
        lines = csvfile.readlines()
        for l in lines[1:]:
            ll = l.split(';')
            rank_id = int(ll[0])
            mpi_type = int(ll[1])
            task_id = int(ll[6])
            time_app = ll[8]
            time_slack = ll[9]
            time_mpi = ll[10]
            wrtxt += l.rstrip()
            if mpi_type in prev_rank_id:
                if rank_id != prev_rank_id[mpi_type]:
                    prev_rank_id[mpi_type] = rank_id
                    prev_task_id[mpi_type] = task_id
                    prev_time_app[mpi_type] = -1
                    prev_time_slack[mpi_type] = -1
                    prev_time_mpi[mpi_type] = -1
            else:
                prev_rank_id[mpi_type] = rank_id
                prev_task_id[mpi_type] = task_id
                prev_time_app[mpi_type] = -1
                prev_time_slack[mpi_type] = -1
                prev_time_mpi[mpi_type] = -1

            wrtxt += ';{};{};{}'.format(prev_time_app[mpi_type], 
                    prev_time_slack[mpi_type], prev_time_mpi[mpi_type])
            prev_rank_id[mpi_type] = rank_id
            prev_task_id[mpi_type] = task_id
            prev_time_app[mpi_type] = time_app
            prev_time_slack[mpi_type] = time_slack
            prev_time_mpi[mpi_type] = time_mpi
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

'''
Function that loads the previously computed prediction errors (see
predict_duration_single_application()) and plots them
'''
def plot_hist_errs_multi_benchmark(err_type):
    print("Time app hist")
    stat_res_app = {}
    ymax = 0
    directory = os.fsencode(stats_res_dir)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("_app.pickle"): 
            pickle_file = stats_res_dir + '/' + filename
            b = filename.split('_app')[0]
            with open(pickle_file, 'rb') as handle:
                stat_res_app[b] = pickle.load(handle)
            if err_type == 'SP':
                plt.hist(stat_res_app[b]['SP_ABS_ERRORS'], bins=100, label=b)
            elif err_type == 'P':
                plt.hist(stat_res_app[b]['P_ABS_ERRORS'], bins=100, label=b)
    plt.xlabel('Error (%)')
    plt.ylabel('# Samples')
    ax.tick_params(axis='y', labelsize=8)
    plt.legend(loc='upper right', prop={'size': 8})
    if err_type == 'SP':
        plotname = plot_dir + '/app_hist_SMAPEerr_multiBench.png'
    elif err_type == 'P':
        plotname = plot_dir + '/app_hist_MAPEerr_multiBench.png'
    plt.savefig(plotname)

    print("Time slack hist")
    stat_res_slack = {}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("_slack.pickle"): 
            pickle_file = stats_res_dir + '/' + filename
            b = filename.split('_slack')[0]
            with open(pickle_file, 'rb') as handle:
                stat_res_slack[b] = pickle.load(handle)
            if err_type == 'SP':
                plt.hist(stat_res_slack[b]['SP_ABS_ERRORS'], bins=200, label=b)
            elif err_type == 'P':
                plt.hist(stat_res_slack[b]['P_ABS_ERRORS'], bins=200, label=b)
    plt.xlabel('Error (%)')
    plt.ylabel('# Samples')
    plt.legend(loc='upper right', prop={'size': 8})
    ax.tick_params(axis='y', labelsize=8)
    if err_type == 'SP':
        plotname = plot_dir + '/slack_hist_SMAPEerr_multiBench.png'
    elif err_type == 'P':
        plotname = plot_dir + '/slack_hist_MAPEerr_multiBench.png'
    plt.savefig(plotname)

    print("Time mpi hist")
    stat_res_mpi = {}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("_mpi.pickle"): 
            pickle_file = stats_res_dir + '/' + filename
            b = filename.split('_mpi')[0]
            with open(pickle_file, 'rb') as handle:
                stat_res_mpi[b] = pickle.load(handle)
            if err_type == 'SP':
                plt.hist(stat_res_mpi[b]['SP_ABS_ERRORS'], bins=200, label=b)
            elif err_type == 'P':
                plt.hist(stat_res_mpi[b]['P_ABS_ERRORS'], bins=200, label=b)
    plt.xlabel('Error (%)')
    plt.ylabel('# Samples')
    plt.legend(loc='upper right', prop={'size': 8})
    ax.tick_params(axis='y', labelsize=8)
    if err_type == 'SP':
        plotname = plot_dir + '/mpi_hist_SMAPEerr_multiBench.png'
    elif err_type == 'P':
        plotname = plot_dir + '/mpi_hist_MAPEerr_multiBench.png'
    plt.savefig(plotname)

    print("---- Per benchmark ----")
    for b in stat_res_app.keys():
        print("\t{}".format(b))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if err_type == 'SP':
            plt.hist(stat_res_app[b]['SP_ABS_ERRORS'], bins=200, label='App')
            plt.hist(stat_res_slack[b]['SP_ABS_ERRORS'], bins=200,label='Slack')
            plt.hist(stat_res_mpi[b]['SP_ABS_ERRORS'], bins=200, label='MPI')
        elif err_type == 'P':
            plt.hist(stat_res_app[b]['P_ABS_ERRORS'], bins=200, label='App')
            plt.hist(stat_res_slack[b]['P_ABS_ERRORS'], bins=200,label='Slack')
            plt.hist(stat_res_mpi[b]['P_ABS_ERRORS'], bins=200, label='MPI')
        plt.xlabel('Error (%)')
        plt.ylabel('# Samples')
        plt.legend(loc='upper right', prop={'size': 8})
        ax.tick_params(axis='y', labelsize=8)
        if err_type == 'SP':
            plotname = plot_dir + '/{}_hist_SMAPEerr_multiTarget.png'.format(b)
        elif err_type == 'P':
            plotname = plot_dir + '/{}_hist_MAPEerr_multiTarget.png'.format(b)
        plt.savefig(plotname)

'''
Plot the importance of each feature
- one plot for each target
'''
def plot_feat_importances(fname):

    #print("Time app")
    #fig = plt.figure()
    #app_feats = []
    #app_means = []
    #app_stds = []
    #for k in feature_importances_app.keys():
    #    app_feats.append(k)
    #    app_means.append(np.mean(np.asarray(feature_importances_app[k])))
    #    app_stds.append(np.std(np.asarray(feature_importances_app[k])))
    #plt.errorbar(app_feats, app_means, yerr=app_stds, fmt='o')
    #plt.xlabel('Feature')
    #plt.ylabel('Relative Importance')
    #plt.ylim(bottom=0)
    #plt.show()

    #print("Time slack")
    #fig = plt.figure()
    #slack_feats = []
    #slack_means = []
    #slack_stds = []
    #for k in feature_importances_slack.keys():
    #    slack_feats.append(k)
    #    slack_means.append(np.mean(np.asarray(feature_importances_slack[k])))
    #    slack_stds.append(np.std(np.asarray(feature_importances_slack[k])))
    #plt.errorbar(slack_feats, slack_means, yerr=slack_stds, fmt='o')
    #plt.xlabel('Feature')
    #plt.ylabel('Relative Importance')
    #plt.ylim(bottom=0)
    #plt.show()

    #print("Time mpi")
    #fig = plt.figure()
    #mpi_feats = []
    #mpi_means = []
    #mpi_stds = []
    #for k in feature_importances_mpi.keys():
    #    mpi_feats.append(k)
    #    mpi_means.append(np.mean(np.asarray(feature_importances_mpi[k])))
    #    mpi_stds.append(np.std(np.asarray(feature_importances_mpi[k])))
    #plt.errorbar(mpi_feats, mpi_means, yerr=mpi_stds, fmt='o')
    #plt.xlabel('Feature')
    #plt.ylabel('Relative Importance')
    #plt.ylim(bottom=0)
    #plt.show()

    app_feats = []
    app_means = []
    app_stds = []
    slack_feats = []
    slack_means = []
    slack_stds = []
    mpi_feats = []
    mpi_means = []
    mpi_stds = []
    for k in feature_importances_app.keys():
        app_feats.append(k)
        app_means.append(np.mean(np.asarray(feature_importances_app[k])))
        app_stds.append(np.std(np.asarray(feature_importances_app[k])))
        slack_feats.append(k)
        slack_means.append(np.mean(np.asarray(feature_importances_slack[k])))
        slack_stds.append(np.std(np.asarray(feature_importances_slack[k])))
        mpi_feats.append(k)
        mpi_means.append(np.mean(np.asarray(feature_importances_mpi[k])))
        mpi_stds.append(np.std(np.asarray(feature_importances_mpi[k])))
    ind = np.arange(len(app_feats))
    ax = plt.subplot(111)
    ax.bar(ind - .2, app_means, yerr=app_stds, width=0.2, color='b', alpha=.6,
            label='App')
    ax.bar(ind, slack_means, yerr=slack_stds, width=0.2, color='r', alpha=.6,
            label='Slack')
    ax.bar(ind + .2, slack_means, yerr=mpi_stds, width=0.2, color='g', alpha=.6
            , label='MPI')
    ax.set_xticks(ind + .2 / 2)
    ax.set_xticklabels((app_feats))
    plt.legend()
    plt.xlabel('Features')
    plt.ylabel('Relative Importance')
    plt.ylim(bottom=0)
    plt.show()

def main(argv):
    pred_type = int(argv[0])
    fname = argv[1]

    if pred_type == 0:
        predict_duration_single_application(fname)
    elif pred_type == 1:
        create_new_csv_with_prev(fname)
    elif pred_type == 2:
        plot_hist_errs_multi_benchmark(fname)
    elif pred_type == 3:
        create_new_csv_with_prev_checkMPItype(fname)
    elif pred_type == 4:
        plot_feat_importances(fname)


if __name__ == '__main__':
    main(sys.argv[1:])

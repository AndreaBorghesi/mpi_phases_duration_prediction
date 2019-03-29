'''
Misc util function used by several other scripts

Author: Andrea Borghesi
    University of Bologna
Date: 20180712
'''

import os
import sys
import math
from decimal import *
import collections
import operator
import json
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer, MinMaxScaler
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn_pandas import DataFrameMapper
import time
import re
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import numpy as np
import scipy.stats as stats
import ast
import numpy as np
np.seterr(all='warn')
import warnings
from pyDOE import *
from eli5.permutation_importance import get_score_importances
import tensorflow as tf

seed = 42

very_large_error = 1000.00

'''
Embedded feature importances for several benchmarks.
These were computed (see predict*.py scripts) using the whole data set and
standard parameters.
TODO: Temporary solution: this should be computed on the fly.
'''
'''
CLASSIFIER
'''
clf_feat_importances = { 
        'DT': {'FWT': [0.521, 0.479], 
            'dwt': [0.26, 0.241, 0.129, 0.006, 0.003, 0.163, 0.197],
            'correlation': [0.140, 0.002, 0.281, 0.000, 0.260, 0.314, 0.001],
            'BlackScholes': [0.004, 0.181, 0.045, 0.155, 0.100, 0.005, 0.001, 
                0.130, 0.071, 0.002, 0.051, 0.074, 0.007, 0.166,0.001],
            'Jacobi': [0.008, 0.005, 0.009, 0.007, 0.003, 0.011, 0.087, 0.074, 
                0.091, 0.138, 0.017, 0.014, 0.008, 0.006, 0.158, 0.055, 0.047, 
                0.069, 0.118, 0.008, 0.006, 0.011, 0.008, 0.015, 0.012]},

        'NN': {'FWT': [0.471, 0.528], 
            'dwt': [0.175, 0.197, 0.197, 0.012, 0.014, 0.203, 0.198],
            'correlation': [0.152, 0.024, 0.246, 0.018, 0.278, 0.252, 0.025],
            'BlackScholes': [0.029, 0.106, 0.050, 0.108, 0.099, 0.036, 0.034, 
                0.123, 0.072, 0.032, 0.062, 0.073, 0.042, 0.094, 0.032],
            'Jacobi': [0.020, 0.016, 0.020, 0.020, 0.021, 0.018, 0.077, 0.080, 
                0.071, 0.070, 0.017, 0.014, 0.023, 0.017, 0.073, 0.073, 0.084,
                0.070, 0.070, 0.027, 0.023, 0.025, 0.023, 0.018, 0.018]}}
'''
REGRESSOR
'''
baseline_regr_feat_importances = { 
        'DT': {'FWT': [0.511, 0.489], 
            'saxpy': [0.000, 0.538, 0.462],
            'convolution': [0.237, 0.220, 0.299, 0.244],
            'dwt': [0.259, 0.189, 0.155, 0.026, 0.014, 0.133, 0.223],
            'correlation': [0.206, 0.002, 0.215, 0.0500, 0.245, 0.227,  0.054],
            'BlackScholes': [0.035, 0.092, 0.042, 0.087, 0.085, 0.040, 0.000, 
                0.212, 0.051, 0.012, 0.066, 0.156, 0.028, 0.092, 0.000],
            'Jacobi': [0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.095, 0.101, 
                0.097, 0.099, 0.007, 0.001, 0.004, 0.000, 0.126, 0.126, 0.089, 
                0.141, 0.112, 0.000, 0.000, 0.000,  0.000, 0.000, 0.000]},

        'NN': {'FWT': [0.554, 0.446], 
            'saxpy': [0.000, 0.469, 0.531],
            'convolution': [0.257, 0.177, 0.325, 0.241],
            'dwt': [0.258, 0.170, 0.144, 0.029, 0.040, 0.206, 0.152],
            'correlation': [0.252, 0.000, 0.019, 0.201, 0.041, 0.015, 0.471],
            'BlackScholes': [0.133, 0.051, 0.152, 0.079, 0.003, 0.193, 0.018, 
                0.129, 0.044, 0.035, 0.039, 0.014, 0.053, 0.058, 0.000],
            'Jacobi': [0.000, 0.000, 0.000, 0.000, 0.005, 0.010, 0.093, 0.147, 
                0.119, 0.122, 0.039, 0.000, 0.000, 0.009, 0.097, 0.039, 0.109, 
                0.105, 0.116, 0.000, 0.000, 0.003, 0.009, 0.002, 0.007]}}


'''
Obtain experiment configuration via latin hipercube sampling
'''
def configs_LHS(nVar, nSamples, max_nbit=53, min_nbit=4):
    samples = lhs(nVar, nSamples, criterion='center')
    var_range = max_nbit - min_nbit
    configs = []
    for s in range(nSamples):
        configs.append([])
    for i in range(len(samples)):
        for var in samples[i]:
            configs[i].append(int(min_nbit + var_range * var))
    return configs

def run_greedy_search(binary_name, err_rate):
    cmds = ['mpirun', '-np', str(n_mpi_procs), greedy_search_script,
            str(seed), binary_name, str(refine), err_rate]
    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    output, err = p.communicate()
    final_config = ast.literal_eval((output.split('\n')[-2]).split(';')[0])
    min_config = ast.literal_eval((output.split('\n')[-2]).split(';')[1])
    return final_config, min_config

def run_greedy_search_opt(binary_name, err_rate, benchmark):
    cmds = ['mpirun', '-np', str(n_mpi_procs), greedy_search_opt_script,
            str(seed), binary_name, str(refine), err_rate, benchmark]
    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    output, err = p.communicate()
    final_config = ast.literal_eval((output.split('\n')[-2]).split(';')[0])
    min_config = ast.literal_eval((output.split('\n')[-2]).split(';')[1])
    return final_config, min_config

def read_file(filename):
    with open(filename, 'r') as rf:
        lines = rf.readlines()
    error_ratios_exps = map(int, lines[0].split(',')[:-1])

    exps_res = {}
    lines = lines[1:]
    for i in range(len(lines)):
        exps_res[i] = map(int, lines[i].split(',')[:-1])
        
    return error_ratios_exps, exps_res

def read_multiple_result_file(basedir, benchmark):
    multi_exp_res = {}
    for filename in os.listdir(basedir):
        if filename.startswith('exp_results_' + benchmark
                ) and filename.endswith('.pickle'):
            with open(basedir + filename, 'rb') as handle:
                exp_res = pickle.load(handle)
            multi_exp_res.update(exp_res)
    return multi_exp_res

def read_result_dict(filename):
    with open(filename, 'r') as rf:
        lines = rf.readlines()
    error_ratios_exps = []
    error_ratios = []
    exps_res = {}
    exps_res_dicts = {}
    configs = []
    times = []
    error_actuals = []
    for l in lines:
        er_exp = int(l.split(';')[0])
        er = l.split(';')[1]
        exp_dict = ast.literal_eval(l.split(';')[2])
        exps_res_dicts[er_exp] = exp_dict
        error_ratios_exps.append(er_exp)
        error_ratios.append(er)
        for i in range(len(exp_dict['var_bits'])):
            if i not in exps_res:
                exps_res[i] = [exp_dict['var_bits'][i]]
            else:
                exps_res[i].append(exp_dict['var_bits'][i])
        if len(l.split(';')) > 3:
            error_actual = float(l.split(';')[3])
        else:
            error_actual = -1.0
        error_actuals.append(error_actual)
        configs.append(exp_dict['var_bits'])
        times.append(exp_dict['time'])
    return (error_ratios_exps, error_ratios, exps_res_dicts, 
            exps_res, configs, times, error_actuals)

def run_program(program, target_result):
    output = subprocess.Popen([program, '%s'%(seed)],
                              stdout=subprocess.PIPE).communicate()[0]
    floating_result = parse_output(output.decode('utf-8'))
    return check_output(floating_result, target_result)

def check_all_zeros(result):
    for i in range(len(result)):
        if result[i] == 0.00:
            return False
    return True

def check_output(floating_result, target_result):
    if len(floating_result) != len(target_result):
        print('Error: floating result len %s while target_result len %s'
                % (len(floating_result), len(target_result)))
        return very_large_error

    if check_all_zeros(floating_result) != check_all_zeros(target_result):
        return very_large_error

    signal_sqr = 0.00
    error_sqr = 0.00
    sqnr = 0.00
    for i in range(len(floating_result)):
        # if floating_result[i] == 0, check_output returns 1: this is an 
        # unwanted behaviour
        if floating_result[i] == 0.00:
            continue    # mmmhhh, TODO: fix this in a smarter way

        # if there is even just one inf in the result list, we assume that
        # for the given configuration the program did not run properly
        if str(floating_result[i]) == 'inf':
            return 'Nan'

        signal_sqr = target_result[i] ** 2
        error_sqr = (floating_result[i] - target_result[i]) ** 2
        temp = 0.00
        if error_sqr != 0.00:
            temp = signal_sqr / error_sqr
        if temp != 0:
            temp = 1.0 / temp
        if temp > sqnr:
            sqnr = temp;

    return sqnr

def parse_output(line):
    list_target = []
    line.replace(' ', '')
    line.replace('\n', '')

    # remove unexpected space
    array = line.split(',')

    for target in array:
        try:
            if len(target) > 0 and target != '\n':
                list_target.append(float(target))
        except:
            continue

    return list_target

def write_conf(conf_file, config):
    conf_string = ''
    for i in config:
        conf_string += str(i) + ','
    with open(conf_file, 'w') as write_file:
        write_file.write(conf_string)

def read_target(target_file):
    # format a1,a2,a3...
    list_target = []
    with open(target_file) as conf_file:
        for line in conf_file:
            line.replace(' ', '')

            # remove unexpected space
            array = line.split(',')
            for target in array:
                try:
                    if len(target) > 0 and target != '\n':
                        list_target.append(float(target))
                except:
                    print('Failed to parse target file')
    return list_target

def read_conf(conf_file_name):
    # format a1,a2,a3,...
    list_argument = []
    with open(conf_file_name) as conf_file:
        for line in conf_file:
            line.replace(' ', '')

            # remove unexpected space
            array = line.split(',')
            for argument in array:
                try:
                    if len(argument) > 0 and argument != '\n':
                        list_argument.append(int(argument))
                except:
                    print('Failed to parse target file')
    return list_argument

'''
Drop unused features and rows with NaN
'''
def drop_stuff(df, features_to_be_dropped):
    for fd in features_to_be_dropped:
        if fd in df:
            del df[fd]
    new_df = df.dropna(axis=0, how='all')
    new_df = new_df.dropna(axis=1, how='all')
    new_df = new_df.fillna(0)
    return new_df

'''
Pre-process input data.
Encode the categorical features
'''
def preprocess_noScaling(df, categorical_features, continuous_features):
    for c in categorical_features:
        df = encode_category(df, c)
    return df

'''
Pre-process input data.
Scale continuous features and encode the categorical ones
'''
def preprocess(df, categorical_features, continuous_features, scaler=None):
    if scaler == None:
        scaler = MinMaxScaler(feature_range=(0, 1))
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    for c in categorical_features:
        df = encode_category(df, c)

    return df, scaler

'''
Plot some information about prediction errors
'''
def plot_errors(abs_errors, p_abs_errors, sp_abs_errors, squared_errors):
    fig = plt.figure()
    plt.hist(np.asarray(p_abs_errors, dtype='float'), bins=50)
    plt.ylabel("# Configs")
    plt.xlabel("Error")
    plt.show()

'''
Plot different variable trends
'''
def plot_VS(x, y, xlabel, ylabel, filename=''):
    fig = plt.figure()
    plt.plot(x, y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if filename != '':
        plt.savefig(filename)
    else:
        plt.show()

''''
Plot different variable trends
- multiple benchmarks
- this function has been used also to plot multiple lines unrelated to 
    benchmark (though the name of the variable hasn't changed - I'm lazy)
'''
def plot_VS_multiBenchmark(xs, ys, xlabel, ylabel, filename='', scale=''):
    fig = plt.figure()
    if scale == 'log':
        ax = fig.add_subplot(1, 1, 1)
    for benchmark, x in xs.items():
        y = ys[benchmark]
        if benchmark == 'Real':
            plt.plot(x, y, label=benchmark, lw=2, linestyle='--')
        else:
            plt.plot(x, y, label=benchmark, lw=2)
        if scale == 'log':
            ax.set_yscale('log')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    #plt.legend(loc='lower right')
    #plt.legend(loc='upper right')
    plt.legend(loc='upper left')
    if filename != '':
        plt.savefig(filename)
    else:
        plt.show()

'''
Encode categorical values into with one-hot encoding
'''
def encode_category(data_set_in, c):
    if c not in data_set_in:
        return data_set_in
    dummy_cols =  pd.get_dummies(data_set_in[c], dummy_na=False)
    data_set = pd.concat([data_set_in, dummy_cols], axis=1)
    del data_set[c]
    return data_set

'''
Evaluate prediction
'''
def evaluate_predictions(predicted, actual):
    abs_errors = []
    p_abs_errors = []
    sp_abs_errors = []
    squared_errors = []
    underest_count = 0
    overest_count = 0

    for i in range(len(predicted)):
        abs_errors.append(abs(predicted[i] - actual[i]))
        squared_errors.append((predicted[i] - actual[i])*
            (predicted[i] - actual[i]))
        if actual[i] != 0:
            p_abs_errors.append((abs(predicted[i]-actual[i]))* 
                    100 / abs(actual[i]))
        sp_abs_errors.append((abs(predicted[i]-actual[i])) * 100 / 
            abs(predicted[i] + actual[i]))
        if predicted[i] - actual[i] > 0:
            overest_count += 1
        elif predicted[i] - actual[i] < 0:
            underest_count += 1

    MAE = Decimal(np.mean(np.asarray(abs_errors)))
    MAPE = Decimal(np.mean(np.asarray(p_abs_errors)))
    SMAPE = Decimal(np.nanmean(np.asarray(sp_abs_errors)))
    MSE = Decimal(np.mean(np.asarray(squared_errors)))
    RMSE = Decimal(math.sqrt(MSE))
    R2 = r2_score(actual, predicted)
    SK_MAE = mean_absolute_error(actual, predicted)
    MedAE = median_absolute_error(actual, predicted)
    SK_MSE = mean_squared_error(actual, predicted)
    EV = explained_variance_score(actual, predicted)

    stats_res = {}
    stats_res["MAE"] = MAE
    stats_res["MSE"] = MSE
    stats_res["RMSE"] = RMSE
    stats_res["MAPE"] = MAPE
    stats_res["SMAPE"] = SMAPE
    stats_res["ABS_ERRORS"] = abs_errors
    stats_res["P_ABS_ERRORS"] = p_abs_errors
    stats_res["SP_ABS_ERRORS"] = sp_abs_errors
    stats_res["SQUARED_ERRORS"] = squared_errors
    stats_res["R2"] = R2
    stats_res["MedAE"] = MedAE
    stats_res["EV"] = EV
    stats_res["abs_errs"] = abs_errors
    stats_res["p_abs_errs"] = p_abs_errors
    stats_res["sp_abs_errs"] = sp_abs_errors
    stats_res["squared_errs"] = squared_errors
    stats_res["accuracy"] = 100 - abs(MAPE)
    stats_res["underest_count"] = underest_count
    stats_res["overest_count"] = overest_count
    stats_res["underest_ratio"] = underest_count / len(predicted)
    stats_res["overest_ratio"] = overest_count / len(predicted)

    return stats_res

'''
Evaluate prediction
- works with NN output
'''
def evaluate_predictions_NN(predicted, actual):
    abs_errors = []
    p_abs_errors = []
    sp_abs_errors = []
    squared_errors = []
    underest_count = 0
    overest_count = 0

    #print(predicted)
    #print(actual)

    for i in range(len(predicted)):
        abs_errors.append(abs(predicted[i][0] - actual[i]))
        squared_errors.append((predicted[i][0] - actual[i])*
            (predicted[i][0] - actual[i]))
        if actual[i] != 0:
            p_abs_errors.append((abs(predicted[i][0]-actual[i]))* 
                    100 / abs(actual[i]))
        sp_abs_errors.append((abs(predicted[i][0]-actual[i])) * 100 / 
            abs(predicted[i][0] + actual[i]))
        if predicted[i] - actual[i] > 0:
            overest_count += 1
        elif predicted[i] - actual[i] < 0:
            underest_count += 1

    MAE = Decimal(np.mean(np.asarray(abs_errors)))
    MAPE = Decimal(np.mean(np.asarray(p_abs_errors)))
    SMAPE = Decimal(np.nanmean(np.asarray(sp_abs_errors)))
    MSE = Decimal(np.mean(np.asarray(squared_errors)))
    RMSE = Decimal(math.sqrt(MSE))
    R2 = r2_score(actual, predicted)
    SK_MAE = mean_absolute_error(actual, predicted)
    MedAE = median_absolute_error(actual, predicted)
    SK_MSE = mean_squared_error(actual, predicted)
    EV = explained_variance_score(actual, predicted)

    stats_res = {}
    stats_res["MAE"] = MAE
    stats_res["MSE"] = MSE
    stats_res["RMSE"] = RMSE
    stats_res["MAPE"] = MAPE
    stats_res["SMAPE"] = SMAPE
    stats_res["ABS_ERRORS"] = abs_errors
    stats_res["P_ABS_ERRORS"] = p_abs_errors
    stats_res["SP_ABS_ERRORS"] = sp_abs_errors
    stats_res["SQUARED_ERRORS"] = squared_errors
    stats_res["R2"] = R2
    stats_res["MedAE"] = MedAE
    stats_res["EV"] = EV
    stats_res["abs_errs"] = abs_errors
    stats_res["p_abs_errs"] = p_abs_errors
    stats_res["sp_abs_errs"] = sp_abs_errors
    stats_res["squared_errs"] = squared_errors
    stats_res["accuracy"] = 100 - abs(MAPE)
    stats_res["underest_count"] = underest_count
    stats_res["overest_count"] = overest_count
    stats_res["underest_ratio"] = underest_count / len(predicted)
    stats_res["overest_ratio"] = overest_count / len(predicted)

    return stats_res

'''
Evaluate binary classification results
- works with NN output
'''
def evaluate_binary_classification_NN(predicted, actual):
    pred_classes = []
    for i in range(len(predicted)):
        if abs(predicted[i][0]) < 0.5:
            pred_classes.append(0)
        else:
            pred_classes.append(1)

    precision_S, recall_S, fscore_S, xyz = precision_recall_fscore_support(
            actual, pred_classes, average='binary', pos_label=0)
    precision_L, recall_L, fscore_L, xyz = precision_recall_fscore_support(
            actual, pred_classes, average='binary', pos_label=1)
    precision_W, recall_W, fscore_W, xyz = precision_recall_fscore_support(
            actual, pred_classes, average='weighted')

    stats_res = {}
    stats_res['small_error_precision'] = precision_S
    stats_res['small_error_recall'] = recall_S
    stats_res['small_error_fscore'] = fscore_S
    stats_res['large_error_precision'] = precision_L
    stats_res['large_error_recall'] = recall_L
    stats_res['large_error_fscore'] = fscore_L
    stats_res['precision'] = precision_W
    stats_res['recall'] = recall_W
    stats_res['fscore'] = fscore_W

    #print(predicted)
    ##print(tf.convert_to_tensor(pred_classes))
    #print(pred_classes)
    #print(set(pred_classes))
    #print(actual.tolist())
    #print(fscore_W)
    #sys.exit()
    return stats_res

'''
Evaluate binary classification results
- works with DT output
'''
def evaluate_binary_classification_DT(predicted, actual):
    pred_classes = []
    for i in range(len(predicted)):
        if predicted[i] < 0.5:
            pred_classes.append(0)
        else:
            pred_classes.append(1)

    precision_S, recall_S, fscore_S, xyz = precision_recall_fscore_support(
            actual, pred_classes, average='binary', pos_label=0)
    precision_L, recall_L, fscore_L, xyz = precision_recall_fscore_support(
            actual, pred_classes, average='binary', pos_label=1)
    precision_W, recall_W, fscore_W, xyz = precision_recall_fscore_support(
            actual, pred_classes, average='weighted')

    stats_res = {}
    stats_res['small_error_precision'] = precision_S
    stats_res['small_error_recall'] = recall_S
    stats_res['small_error_fscore'] = fscore_S
    stats_res['large_error_precision'] = precision_L
    stats_res['large_error_recall'] = recall_L
    stats_res['large_error_fscore'] = fscore_L
    stats_res['precision'] = precision_W
    stats_res['recall'] = recall_W
    stats_res['fscore'] = fscore_W

    return stats_res

'''
Analyse features importance
- works only with RF
'''
def eval_feature_importance(model, X):
    print("Features importance: %s " % model.feature_importances_)
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
            axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], 
            importances[indices[f]]))
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
                   color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

'''
Load data needed in order to create a prediction model
'''
def load_data_for_prediction(benchmark, benchmarks_home, data_set_dir, 
        data_set_dir_davide, benchmark_nVar, min_nbit, max_nbit):
    benchmark_dir = benchmarks_home + '/' + benchmark + '/'

    nVar = benchmark_nVar[benchmark]
    var_ranges = {}
    for i in range(nVar):
        var_ranges[i] = [x for x in range(min_nbit, max_nbit+1)]

    original_config = [max_nbit for i in range(nVar)]

    if benchmark == 'BlackScholes' or benchmark == 'Jacobi':
        single_file = False
    else:
        single_file = True

    if single_file:    # all experiments results are in the same file
        nSamples = 10000
        if benchmark == 'FWT':
            nSamples = 20000

        pickle_file = data_set_dir + "exp_results_" + benchmark + "_" + str(
                nSamples) + ".pickle"

        if os.path.isfile(pickle_file):
            with open(pickle_file, 'rb') as handle:
                exp_res = pickle.load(handle)
        else:          
            print('Data set not available for benchmark %s' % benchmark)
            return  [], [], [], []

    else:              # results are split in different files
        exp_res = read_multiple_result_file(data_set_dir_davide, 
                benchmark)

    errors = []
    configs = []
    vars_vals = {}
    data_dict = {}
    j = 0
    for k in exp_res.keys():
        config, error = exp_res[k]
        configs.append(config)
        errors.append(error)
        data_dict[j] = {'error': error}
        for i in range(len(config)):
            data_dict[j]['var_%s' % i] = config[i]
            if i in vars_vals:
                vars_vals[i].append(config[i])
            else:
                vars_vals[i] = [config[i]]
        j += 1
    return errors, configs, vars_vals, data_dict

'''
Evaluate solution found by optimizer.
- run the actual program
- check whether the generated error is smaller than the desired one
'''
def check_solution(benchmark, opt_config, trgt_error_ratio, benchmarks_home, 
        binary_map, large_error_threshold, benchmark_nVar, max_nbit, min_nbit):
    benchmark_dir = benchmarks_home + '/' + benchmark + '/'
    program = benchmark_dir + binary_map[benchmark] + '.sh'
    config_file = benchmark_dir + 'config_file.txt'
    target_file = benchmark_dir + 'target.txt'
    target_result = read_target(target_file)
    nVar = benchmark_nVar[benchmark]
    original_config = [max_nbit for i in range(nVar)]
    write_conf(config_file, opt_config)
    error = run_program(program, target_result)
    # write back original config
    write_conf(config_file, original_config)
    is_error_se_trgt = (error <= trgt_error_ratio)
    if error < large_error_threshold:
        error_class = 0
    else:
        error_class = 1
    return error, is_error_se_trgt, error_class

'''
Evaluate solution found by optimizer.
- run the actual program
'''
def run_program_withConf(benchmark, opt_config, benchmarks_home, 
        binary_map, benchmark_nVar, max_nbit, min_nbit):
    benchmark_dir = benchmarks_home + '/' + benchmark + '/'
    program = benchmark_dir + binary_map[benchmark] + '.sh'
    config_file = benchmark_dir + 'config_file.txt'
    target_file = benchmark_dir + 'target.txt'
    target_result = read_target(target_file)
    nVar = benchmark_nVar[benchmark]
    original_config = [max_nbit for i in range(nVar)]
    write_conf(config_file, opt_config)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            error = run_program(program, target_result)
        #except eWarning as Exception:
        except Warning:
            print("<<<<<<<<<<<<<<<<<<<< EXCEPPPPPP")
    # write back original config
    write_conf(config_file, original_config)
    return error

'''
Trace statistics
'''
def trace_stats(sol_stats, tracefile):
    if sol_stats == None:
        stats_str = '=======================================================\n'
        with open(tracefile, 'w+') as write_file:
            write_file.write(stats_str)
    else:
        stats_str='{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11}\n'.format(
                sol_stats['config'], sol_stats['delta_config'], 
                sol_stats['error'], sol_stats['delta_error'],
                sol_stats['error_capped'], sol_stats['error_pred'], 
                sol_stats['delta_error_pred'], sol_stats['error_log'],
                sol_stats['delta_error_log'], sol_stats['error_pred_log'],
                sol_stats['delta_error_pred_log'], sol_stats['cost'])
        with open(tracefile, 'a') as write_file:
            write_file.write(stats_str)

'''
Trace statistics
- compact format (human readable)
'''
def trace_stats_compact(sol_stats, tracefile):
    if sol_stats == None:
        stats_str = '=======================================================\n'
        with open(tracefile, 'w+') as write_file:
            write_file.write(stats_str)
    else:
        sstr = ('Conf {0}; Conf delta {1}; Error {2:.3f}; Error Pred {3:.3f}; '
            'Error log {4:.3f}; Error Pred Log {5:.3f}; Sol Cost {6}\n'.format(
            sol_stats['config'], sol_stats['delta_config'], sol_stats['error'], 
            sol_stats['error_pred'], sol_stats['error_log'], 
            sol_stats['error_pred_log'], sol_stats['cost']))
        with open(tracefile, 'a') as write_file:
            write_file.write(sstr)

'''
Pretty print trace statistics
'''
def print_trace_stats(sol_stats):
    print('[sol_stats] Conf {0}; Conf delta {1}; Error {2}; Error Pred {3}; '
            'Error log {4:.3f}; Error Pred Log {5:.3f}; Sol Cost {6}'.format(
            sol_stats['config'], sol_stats['delta_config'], sol_stats['error'], 
            sol_stats['error_pred'], sol_stats['error_log'], 
            sol_stats['error_pred_log'], sol_stats['cost']))

'''
Return classifier features importances (those embedded in the code)
'''
def get_embedded_clf_feat_importances(benchmark, clf_type):
    if clf_type != 'NN' and clf_type != 'DT':  # only types accepted now
        return None
    else:
        return clf_feat_importances[clf_type][benchmark]

'''
Function that returns the feature importances of the model passed as argument
IN: ML model 
IN: model type; string, allowed values: NN, DT
IN: training data (as tensor)
IN: training target (as tensor)
OUT: a vector containing the features importances

Example (called from another script): 
    fi = my_util.get_features_importance(model, 'NN', test_data_tensor,
            test_target_tensor)
'''
def get_features_importance(model, model_type, X, y):
    if model_type == 'DT':
        return model.feature_importances_

    elif model_type == 'NN':
        def score(X, y):
            acc = model.evaluate(X, y, verbose=1)
            return acc
        base_score, score_decreases = get_score_importances(score, X, y)
        feature_importances = np.mean(score_decreases, axis=0)
        feat_sum = sum(feature_importances)
        feature_importances_norm = [fi / feat_sum for fi in
                feature_importances]
        return feature_importances_norm

    else:
        print('Unsupported model type')
        return None


'''
Extract a subset of the whole data set 
- Ideally we want to obtain a subset that resembles the whole data set
'''
def select_data_subset(df_full, size, large_error_threshold):
    # large errors are a big problem; we want to have them in our training set
    # at least with the same proportion they appear in the whol data set
    # TODO: the creation of the initial training set can be refined
    cnt_large_error_whole_data = len(df_full[
            df_full['error'] >= large_error_threshold])
    large_error_ratio_whole_data = cnt_large_error_whole_data / len(df_full)

    large_error_ratio_train_set = 0
    while large_error_ratio_train_set < large_error_ratio_whole_data:
        df = df_full.sample(size)
        large_error_ratio_train_set = len(df[
                df['error'] >= large_error_threshold]) / len(df)

    return df

'''
Create training and test set for both the classification and the regression
task. This is useful in the active learning settings where we want  both
classifier and regressor to start from the same starting point
- the function take as input the data set size to be drawn and the size of the
  training set; i.e. data set size = 120 and train set size = 100
- the training set is extracted from the larger data set via a fixed split (in
  this way both classifier and regressor have the same training set and test
  set)
''' 
def create_train_test_sets(benchmark, benchmarks_home, data_set_dir,
        data_set_dir_davide, benchmark_nVar, min_nbit, max_nbit,
        value_inPlace_of_inf, errors_close_to_0_threshold,
        large_error_threshold, set_size, train_set_size):
    errors, configs, vars_vals, data_dict = load_data_for_prediction(
            benchmark, benchmarks_home, data_set_dir, data_set_dir_davide, 
            benchmark_nVar, min_nbit, max_nbit)

    df_full = pd.DataFrame.from_dict(data_dict)
    df_full = df_full.transpose()

    '''
    There could be some issues in giving np.inf value to configs with error
    equal to zero.
    '''
    df_error_zero = df_full[(df_full['error'] == 0)]
    df_full = df_full[(df_full['error'] != 0)]

    '''
    The error values are very small (i.e. 1e-10, 1e-40..) and the regressor
    struggles to distinguish between the targets (even after scaling). Since we
    are not interested in the prediction of the error _per se_ but only in the 
    relationships between number of bits assigned to variables and error, we
    can use the -log(error), in order to magnify the distance between target
    values.
    '''
    df_full['log_error'] = -np.log(df_full['error'])

    # artificially set config with no error to large error logs
    df_error_zero['log_error'] = value_inPlace_of_inf
    frames = [df_full, df_error_zero]
    result = pd.concat(frames)
    df_full = result
    df_full = df_full.sample(frac=1).reset_index(drop=True)

    #'''
    #error values extremely close to zero (both negative and positive create 
    #lots of problems. I pretend they do not exist for this test
    #'''
    #if benchmark != 'Jacobi':
    #    df_full = df_full[(df_full['log_error'] <= 
    #        -errors_close_to_0_threshold) | 
    #            (df_full['log_error'] >= errors_close_to_0_threshold)]

    if set_size == 'FULL':
        size = len(df_full)
        df = df_full
    else:
        size = set_size

    df = select_data_subset(df_full, size, large_error_threshold)

    df = df[(df != 0).all(1)]

    def error_class(row):
        if row['error'] >= large_error_threshold:
            return 1
        else:
            return 0
    df['error_class'] = df.apply(error_class, axis=1)
   
    target_regr = df['log_error']
    target_classr = df['error_class']

    #print(df['error'])
    #print(target_regr)
    #print(target_classr)
    #sys.exit()

    del df['error']
    del df['log_error']
    del df['error_class']

    train_data_regr = df[:train_set_size]
    train_data_classr = df[:train_set_size]

    train_target_regr = target_regr[:train_set_size]
    train_target_classr = target_classr[:train_set_size]

    test_data_regr = df[train_set_size:]
    test_data_classr = df[train_set_size:]

    test_target_regr = target_regr[train_set_size:]
    test_target_classr = target_classr[train_set_size:]

    return (train_data_regr, test_data_regr, train_target_regr,
            test_target_regr, train_data_classr, test_data_classr,
            train_target_classr, test_target_classr)


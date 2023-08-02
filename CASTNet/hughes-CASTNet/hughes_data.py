import argparse
import os
import pandas as pd
import operator
from collections import OrderedDict
import datetime
from dateutil import relativedelta
import numpy as np
import csv
import pickle
import sys
import math
from scipy.stats import entropy

## dataset_name = [Cincinnati | Chicago]
## time_unit -> w.r.t days (e.g. time_unit=7 means data aggregated weekly.)
def readData(dataset_name, window_size, lead_time, train_ratio, test_ratio, dist, time_unit=1):

    #prefix = 'CASTNet-master/Data/' + dataset_name + '/'
    prefix = '/Users/jyontika/Desktop/Python/github_hughes/opioid-overdose-models/CASTNet/CASTNet-master/Data/' +dataset_name + '/'
    
    locations_path = prefix + 'locations.txt'
    locations_path = prefix + 'locations.txt'
    distances_path = prefix +  'distances.csv'
    static_features_path = prefix + 'static_features.csv'
    svi_path = prefix + 'svi.pkl'
    od_path = prefix + 'overdose.pkl'

    locations = []
    location_numbers = []
    with open(locations_path, 'rb') as file:
        for line in file:
            line = line.rstrip().decode("utf-8").split("\t")
            location_numbers.append(line[0])
            locations.append(line[1])
    
    ################################################################################################
    static_features_raw = pd.read_csv(static_features_path)
    static_features = np.zeros(shape=(static_features_raw.shape[1], static_features_raw.shape[0]))
    for i in range(0, static_features_raw.shape[1]):
        for j in range(0, static_features_raw.shape[0]):
            static_features[i][j] = static_features_raw.iloc[j][i]
    
    #### Normalize population (male, female, race and poverty) ###
    for i in range(0, static_features.shape[0]):
        static_features[i][1:11] /= static_features[i][0]
        static_features[i][13] /= static_features[i][0]
    
    static_features[:, 0] = np.log(static_features[:, 0])
    income_mean = np.mean(static_features[:, 11])
    income_std = np.std(static_features[:, 11])
    static_features[:, 11] = static_features[:, 11] - income_mean
    static_features[:, 11] /= income_std
    per_capita_mean = np.mean(static_features[:, 12])
    per_capita_std = np.std(static_features[:, 12])
    static_features[:,12] = static_features[:, 12] - per_capita_mean
    static_features[:, 12] /= per_capita_std
    #################################################################################################
    new_static_feature_size = 9
    static_features_new = np.zeros(shape=(static_features_raw.shape[1], new_static_feature_size))
    static_features_new[:, 0] = (static_features[:, 0] - np.mean(static_features[:, 0])) / np.std(static_features[:, 0])
    for i in range(0, static_features_new.shape[0]):
        ## sex-diversity, race_diversity (normalized cross entropy)##
        static_features_new[i][1] = entropy(static_features[i][1:3]) / np.log(2)
        static_features_new[i][2] = entropy(static_features[i][3:11]) / np.log(8)
    static_features_new[:, 3:] = np.copy(static_features[:, 11:])
    static_features = np.copy(static_features_new)
    #################################################################################################
    
    with open(svi_path, 'rb') as file:
        svi = pickle.load(file, encoding='bytes')
    
    #print(svi)
    print(svi.shape)

    with open(od_path, 'rb') as file:
        overdose = pickle.load(file, encoding='bytes')
        
    overdose = np.swapaxes(overdose, 1, 0)
    
    static_feature_size = static_features.shape[1]
    
    num_agg_slots = int(math.ceil(svi.shape[1] / float(time_unit)))
    svi_agg = np.zeros(shape=(svi.shape[0], num_agg_slots, svi.shape[2]))
    overdose_agg = np.zeros(shape=(overdose.shape[0], num_agg_slots))
    for loc in range(0, svi.shape[0]):
        new_time_idx = 0
        for i in range(0, svi.shape[1], time_unit):
            start_idx = i
            end_idx = i + time_unit
            if end_idx > svi.shape[1]:
                end_idx = svi.shape[1]
            
            svi_agg[loc, new_time_idx] = np.sum(svi[loc, start_idx:end_idx], axis=0)
            overdose_agg[loc, new_time_idx] = np.sum(overdose[loc, start_idx:end_idx])
            new_time_idx += 1
            
    
    num_time_slots = svi_agg.shape[1]
    num_days = num_time_slots
    num_train_days = int(round(num_days * train_ratio))
    num_test_days = int(round(num_days * test_ratio))
    num_valid_days = num_days - num_train_days - num_test_days
    
    train_start_index = 0
    valid_start_index = num_train_days - (window_size + lead_time - 1)
    test_start_index = num_train_days + num_valid_days - (window_size + lead_time - 1)
    
    #print train_start_index, valid_start_index
    #print valid_start_index, test_start_index, '(num_valid_samples):', num_valid_days
    #print test_start_index, test_start_index + num_test_days, '(num_test_samples):', num_test_days
    
    distances_row = pd.read_csv(distances_path, header=None)
    distances = distances_row.values
    
    if dist == 'no':
        distances.fill(1.)
    else:
        distances = distances / 1000.
        
        if dist == 'd':
            distances = 1. / (1. + distances)
        elif dist == 'd2':
            distances = 1. / np.square(1. + distances)
        elif dist == 'log':
            distances = 1. / (1. + np.log(distances))
        elif dist == 'root_d':
            distances = 1. / np.sqrt(1. + distances)
    
    
    ####################################################################################################
    ## Normalization of svi svi features #########################################################
    ####################################################################################################
    for i in range(0, svi_agg.shape[2]):
        mean_values = np.mean(svi_agg[:, :num_train_days, i])
        std_values = np.std(svi_agg[:, :num_train_days, i])
        svi_agg[:, :, i] = (svi_agg[:, :, i] - mean_values) / std_values
    ####################################################################################################
    
    train_svi_local = np.zeros(shape=(len(locations) * (num_train_days - (window_size + lead_time - 1)), window_size, svi_agg.shape[2] + 1))
    train_svi_global = np.zeros(shape=(len(locations) * (num_train_days - (window_size + lead_time - 1)), len(locations), window_size, svi_agg.shape[2] + 1))
    train_static = np.zeros(shape=(len(locations) * (num_train_days - (window_size + lead_time - 1)), static_feature_size))
    train_sample_indices = np.zeros(shape=(len(locations) * (num_train_days - (window_size + lead_time - 1)),))
    train_dist = np.ones(shape=(len(locations) * (num_train_days - (window_size + lead_time - 1)), len(locations)))
    train_y = np.zeros(shape=(len(locations) * (num_train_days - (window_size + lead_time - 1)), ))
    
    valid_svi_local = np.zeros(shape=(len(locations) * num_valid_days, window_size, svi_agg.shape[2] + 1))
    valid_svi_global = np.zeros(shape=(len(locations) * num_valid_days, len(locations), window_size, svi_agg.shape[2] + 1))
    valid_static = np.zeros(shape=(len(locations) * num_valid_days, static_feature_size))
    valid_sample_indices = np.zeros(shape=(len(locations) * num_valid_days,))
    valid_dist = np.ones(shape=(len(locations) * num_valid_days, len(locations)))
    valid_y = np.zeros(shape=(len(locations) * num_valid_days, ))
    
    test_svi_local = np.zeros(shape=(len(locations) * num_test_days, window_size, svi_agg.shape[2] + 1))
    test_svi_global = np.zeros(shape=(len(locations) * num_test_days, len(locations), window_size, svi_agg.shape[2] + 1))
    test_static = np.zeros(shape=(len(locations) * num_test_days, static_feature_size))
    test_sample_indices = np.zeros(shape=(len(locations) * num_test_days,))
    test_dist = np.ones(shape=(len(locations) * num_test_days, len(locations)))
    test_y = np.zeros(shape=(len(locations) * num_test_days, ))
    
    
    od_mean = np.mean(overdose_agg[:, 0:num_train_days])
    od_std = np.std(overdose_agg[:, 0:num_train_days])
    
    counter = 0
    for l in range(0, len(locations)):
        for i in range(train_start_index, valid_start_index):
            for l_neighbor in range(0, len(locations)):
                for k in range(0, window_size):
                    train_svi_global[counter][l_neighbor][k] = np.concatenate([svi_agg[l_neighbor][i+k],
                                                                               np.array([(overdose_agg[l_neighbor][i+k] - od_mean) / od_std])])
                    
            for k in range(0, window_size):
                train_svi_local[counter][k] = np.concatenate([svi_agg[l][i+k],
                                                                     np.array([(overdose_agg[l][i+k] - od_mean) / od_std])])
            train_static[counter] = static_features[l]
            train_sample_indices[counter] = l
            train_dist[counter] = distances[l]
            train_y[counter] = overdose_agg[l][i + window_size + lead_time - 1]
            counter += 1
    
    counter = 0
    for l in range(0, len(locations)):
        for i in range(valid_start_index, test_start_index):
            for l_neighbor in range(0, len(locations)):
                for k in range(0, window_size):
                    valid_svi_global[counter][l_neighbor][k] = np.concatenate([svi_agg[l_neighbor][i+k], 
                                                                               np.array([(overdose_agg[l_neighbor][i+k] - od_mean) / od_std])])
            for k in range(0, window_size):
                valid_svi_local[counter][k] = np.concatenate([svi_agg[l][i+k], 
                                                                     np.array([(overdose_agg[l][i+k] - od_mean) / od_std])])
            valid_static[counter] = static_features[l]
            valid_sample_indices[counter] = l
            valid_dist[counter] = distances[l]
            valid_y[counter] = overdose_agg[l][i + window_size + lead_time - 1]
            counter += 1
    
    counter = 0
    for l in range(0, len(locations)):
        for i in range(test_start_index, test_start_index + num_test_days):
            for l_neighbor in range(0, len(locations)):
                for k in range(0, window_size):
                    test_svi_global[counter][l_neighbor][k] = np.concatenate([svi_agg[l_neighbor][i+k], 
                                                                              np.array([(overdose_agg[l_neighbor][i+k] - od_mean) / od_std])])
            for k in range(0, window_size):
                test_svi_local[counter][k] = np.concatenate([svi_agg[l][i+k], 
                                                                    np.array([(overdose_agg[l][i+k] - od_mean) / od_std])])
            test_static[counter] = static_features[l]
            test_sample_indices[counter] = l
            test_dist[counter] = distances[l]
            test_y[counter] = overdose_agg[l][i + window_size + lead_time - 1]
            counter += 1
    
    #print train_svi_global.shape, valid_svi_global.shape, test_svi_global.shape
    #print train_svi_local.shape, valid_svi_local.shape, test_svi_local.shape
    
    train_sample_indices = train_sample_indices.astype(int)
    valid_sample_indices = valid_sample_indices.astype(int)
    test_sample_indices = test_sample_indices.astype(int)
    
    return train_svi_local, train_svi_global, train_static, train_sample_indices, train_dist, train_y, valid_svi_local, valid_svi_global, valid_static, valid_sample_indices, valid_dist, valid_y, test_svi_local, test_svi_global, test_static, test_sample_indices, test_dist, test_y


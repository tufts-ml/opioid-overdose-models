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

## dataset_name = [MA | Chicago]
## time_unit -> w.r.t days (e.g. time_unit=7 means data aggregated weekly.) ## changed this to years (so 1) - jyon.
def readData(castnet_datadir, dataset_name, window_size, lead_time, num_train_years, num_test_years, num_valid_years, dist, time_unit=1):

    prefix = castnet_datadir
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

    static_features_raw = pd.read_csv(static_features_path)
    static_features = np.zeros(shape=(static_features_raw.shape[1], static_features_raw.shape[0]))
    for i in range(0, static_features_raw.shape[1]):
        for j in range(0, static_features_raw.shape[0]):
            static_features[i][j] = static_features_raw.iloc[j][i]
    
    with open(svi_path, 'rb') as file:
        svi = pickle.load(file, encoding='bytes')

    with open(od_path, 'rb') as file:
        overdose = pickle.load(file, encoding='bytes')
        
    overdose = np.swapaxes(overdose, 1, 0)
    
    static_feature_size = static_features.shape[1]
    
    num_years = num_train_years + num_valid_years + num_test_years + (window_size + lead_time - 1)
    print(f'num_years {num_years}')
    # Trim datasets to have only required_number of years
    svi = svi[:,-num_years:, :]
    overdose = overdose[:, -num_years:]
    print(f'Trimmed svi shape{svi.shape}')
     
    train_start_index = 0
    valid_start_index = num_years - num_test_years - num_valid_years - (window_size + lead_time - 1)
    test_start_index = num_years - num_test_years  - (window_size + lead_time - 1)

    print(f'Start indicies {train_start_index}, {valid_start_index}, {test_start_index}')
    
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
    ## Normalization of svi  features #########################################################
    ####################################################################################################
    for i in range(0, svi.shape[2]):
        mean_values = np.mean(svi[:, :num_train_years, i])
        std_values = np.std(svi[:, :num_train_years, i])
        svi[:, :, i] = (svi[:, :, i] - mean_values) / std_values
    ####################################################################################################

    # Print the values of num_train_years, window_size, and lead_time
    print("num_train_years:", num_train_years)
    print("window_size:", window_size)
    print("lead_time:", lead_time)

    # Calculate the difference (num_train_years - (window_size + lead_time - 1))
    diff = num_train_years - (window_size + lead_time - 1)

    # Print the difference
    print("Difference (num_train_years - (window_size + lead_time - 1)):", diff)

    train_svi_local = np.zeros(shape=(len(locations) * (num_train_years), window_size, svi.shape[2] + 1))
    train_svi_global = np.zeros(shape=(len(locations) * (num_train_years), len(locations), window_size, svi.shape[2] + 1))
    train_static = np.zeros(shape=(len(locations) * (num_train_years), static_feature_size))
    train_sample_indices = np.zeros(shape=(len(locations) * (num_train_years),))
    train_dist = np.ones(shape=(len(locations) * (num_train_years), len(locations)))
    train_y = np.zeros(shape=(len(locations) * (num_train_years), ))
    
    valid_svi_local = np.zeros(shape=(len(locations) * num_valid_years, window_size, svi.shape[2] + 1))
    valid_svi_global = np.zeros(shape=(len(locations) * num_valid_years, len(locations), window_size, svi.shape[2] + 1))
    valid_static = np.zeros(shape=(len(locations) * num_valid_years, static_feature_size))
    valid_sample_indices = np.zeros(shape=(len(locations) * num_valid_years,))
    valid_dist = np.ones(shape=(len(locations) * num_valid_years, len(locations)))
    valid_y = np.zeros(shape=(len(locations) * num_valid_years, ))
    
    test_svi_local = np.zeros(shape=(len(locations) * num_test_years, window_size, svi.shape[2] + 1))
    test_svi_global = np.zeros(shape=(len(locations) * num_test_years, len(locations), window_size, svi.shape[2] + 1))
    test_static = np.zeros(shape=(len(locations) * num_test_years, static_feature_size))
    test_sample_indices = np.zeros(shape=(len(locations) * num_test_years,))
    test_dist = np.ones(shape=(len(locations) * num_test_years, len(locations)))
    test_y = np.zeros(shape=(len(locations) * num_test_years, ))
    
    
    od_mean = np.mean(overdose[:, 0:num_train_years])
    od_std = np.std(overdose[:, 0:num_train_years])

    counter = 0
    for l in range(0, len(locations)):
        for i in range(train_start_index, valid_start_index):
            for l_neighbor in range(0, len(locations)):
                for k in range(0, window_size):
                    train_svi_global[counter][l_neighbor][k] = np.concatenate([svi[l_neighbor][i+k],
                                                                               np.array([(overdose[l_neighbor][i+k] - od_mean) / od_std])])
                    
            for k in range(0, window_size):
                train_svi_local[counter][k] = np.concatenate([svi[l][i+k],
                                                                     np.array([(overdose[l][i+k] - od_mean) / od_std])])
            try:
                train_static[counter] = static_features[l]
            except KeyError:
                print("Error: KeyError occurred with l =", l, "and counter =", counter)

            train_sample_indices[counter] = l
            train_dist[counter] = distances[l]
            train_y[counter] = overdose[l][i + window_size + lead_time - 1]
            counter += 1
    
    counter = 0
    for l in range(0, len(locations)):
        for i in range(valid_start_index, test_start_index):
            for l_neighbor in range(0, len(locations)):
                for k in range(0, window_size):
                    valid_svi_global[counter][l_neighbor][k] = np.concatenate([svi[l_neighbor][i+k], 
                                                                               np.array([(overdose[l_neighbor][i+k] - od_mean) / od_std])])
            for k in range(0, window_size):
                valid_svi_local[counter][k] = np.concatenate([svi[l][i+k], 
                                                                     np.array([(overdose[l][i+k] - od_mean) / od_std])])
            valid_static[counter] = static_features[l]
            valid_sample_indices[counter] = l
            valid_dist[counter] = distances[l]
            valid_y[counter] = overdose[l][i + window_size + lead_time - 1]
            counter += 1
    
    counter = 0
    for l in range(0, len(locations)):
        for i in range(test_start_index, test_start_index + num_test_years):
            for l_neighbor in range(0, len(locations)):
                for k in range(0, window_size):
                    test_svi_global[counter][l_neighbor][k] = np.concatenate([svi[l_neighbor][i+k], 
                                                                              np.array([(overdose[l_neighbor][i+k] - od_mean) / od_std])])
            for k in range(0, window_size):
                test_svi_local[counter][k] = np.concatenate([svi[l][i+k], 
                                                                    np.array([(overdose[l][i+k] - od_mean) / od_std])])
            test_static[counter] = static_features[l]
            test_sample_indices[counter] = l
            test_dist[counter] = distances[l]
            test_y[counter] = overdose[l][i + window_size + lead_time - 1]
            counter += 1
    
    #print train_svi_global.shape, valid_svi_global.shape, test_svi_global.shape
    #print train_svi_local.shape, valid_svi_local.shape, test_svi_local.shape
    
    train_sample_indices = train_sample_indices.astype(int)
    valid_sample_indices = valid_sample_indices.astype(int)
    test_sample_indices = test_sample_indices.astype(int)
    
    return train_svi_local, train_svi_global, train_static, train_sample_indices, train_dist, train_y, valid_svi_local, valid_svi_global, valid_static, valid_sample_indices, valid_dist, valid_y, test_svi_local, test_svi_global, test_static, test_sample_indices, test_dist, test_y


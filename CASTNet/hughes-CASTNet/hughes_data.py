import argparse
import os
import pandas as pd
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
def readData(dataset_name, window_size, lead_time, train_ratio, test_ratio, dist, time_unit=7):

    #prefix = 'CASTNet-master/Data/' + dataset_name + '/'
    prefix = '/Users/jyontika/Desktop/Python/Hughes Lab - DIAMONDS/Opioid-Project/Opioid-Project-master/CASTNet-master/Data/' +dataset_name + '/'
    
    locations_path = prefix + 'locations.txt'
    locations_path = prefix + 'locations.txt'
    distances_path = prefix +  'distances.csv'
    static_features_path = prefix + 'static_features.csv'
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
        
    with open(od_path, 'rb') as file:
        overdose = pickle.load(file, encoding='bytes')
        
    overdose = np.swapaxes(overdose, 1, 0)
    
    static_feature_size = static_features.shape[1]
    
    num_agg_slots = int(math.ceil(overdose.shape[1] / float(time_unit)))
    overdose_agg = np.zeros(shape=(overdose.shape[0], num_agg_slots))
    for loc in range(0, overdose.shape[0]):
        new_time_idx = 0
        for i in range(0, overdose.shape[1], time_unit):
            start_idx = i
            end_idx = i + time_unit
            if end_idx > overdose.shape[1]:
                end_idx = overdose.shape[1]
            
            overdose_agg[loc, new_time_idx] = np.sum(overdose[loc, start_idx:end_idx])
            new_time_idx += 1


    num_time_slots = overdose_agg.shape[1]
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
    
    
    
    train_static = np.zeros(shape=(len(locations) * (num_train_days - (window_size + lead_time - 1)), static_feature_size))
    train_sample_indices = np.zeros(shape=(len(locations) * (num_train_days - (window_size + lead_time - 1)),))
    train_dist = np.ones(shape=(len(locations) * (num_train_days - (window_size + lead_time - 1)), len(locations)))
    train_y = np.zeros(shape=(len(locations) * (num_train_days - (window_size + lead_time - 1)), ))
    
    valid_static = np.zeros(shape=(len(locations) * num_valid_days, static_feature_size))
    valid_sample_indices = np.zeros(shape=(len(locations) * num_valid_days,))
    valid_dist = np.ones(shape=(len(locations) * num_valid_days, len(locations)))
    valid_y = np.zeros(shape=(len(locations) * num_valid_days, ))
    

    test_static = np.zeros(shape=(len(locations) * num_test_days, static_feature_size))
    test_sample_indices = np.zeros(shape=(len(locations) * num_test_days,))
    test_dist = np.ones(shape=(len(locations) * num_test_days, len(locations)))
    test_y = np.zeros(shape=(len(locations) * num_test_days, ))
    
    
    od_mean = np.mean(overdose_agg[:, 0:num_train_days])
    od_std = np.std(overdose_agg[:, 0:num_train_days])
    
    counter = 0
    for l in range(0, len(locations)):
        for i in range(train_start_index, valid_start_index):
            train_static[counter] = static_features[l]
            train_sample_indices[counter] = l
            train_dist[counter] = distances[l]
            train_y[counter] = overdose_agg[l][i + window_size + lead_time - 1]
            counter += 1

    counter = 0
    for l in range(0, len(locations)):
        for i in range(valid_start_index, test_start_index):
            valid_static[counter] = static_features[l]
            valid_sample_indices[counter] = l
            valid_dist[counter] = distances[l]
            valid_y[counter] = overdose_agg[l][i + window_size + lead_time - 1]
            counter += 1

    counter = 0
    for l in range(0, len(locations)):
        for i in range(test_start_index, test_start_index + num_test_days):
            test_static[counter] = static_features[l]
            test_sample_indices[counter] = l
            test_dist[counter] = distances[l]
            test_y[counter] = overdose_agg[l][i + window_size + lead_time - 1]
            counter += 1

    

    train_sample_indices = train_sample_indices.astype(int)
    valid_sample_indices = valid_sample_indices.astype(int)
    test_sample_indices = test_sample_indices.astype(int)
    
    return train_static, train_sample_indices, train_dist, train_y, valid_static, valid_sample_indices, valid_dist, valid_y,  test_static, test_sample_indices, test_dist, test_y


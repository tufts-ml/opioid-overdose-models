import os
import sys
from glob import glob

import numpy as np
import pandas as pd
idx = pd.IndexSlice
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import copy

import geopandas as gpd

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

import gpflow
import tensorflow as tf
import sys
code_dir = '/cluster/home/kheuto01/code/zero-inflated-gp/'
sys.path.append(code_dir)
from math import radians, cos, sin, asin, sqrt
from onoffgpf import OnOffSVGP, OnOffLikelihood

import pickle

from math import radians, cos, sin, asin, sqrt

import copy
import sys

import geopandas as gpd
import numpy as np

import gpflow


code_dir = '/cluster/home/kheuto01/code/zero-inflated-gp/'
sys.path.append(code_dir)

code_dir = '/cluster/home/kheuto01/code/opioid-overdose-models/'
sys.path.append(code_dir)

from onoffgpf import OnOffSVGP, OnOffSVGPPoiMC, OnOffLikelihood
gpflow.config.default_float()


from zinf_gp.metrics import normcdf, fixed_top_X


def check_results(data_dir=None, time=None, loc=None,
                  model=None, start_year=None, cov=None,
                  num_inducing=None,
                  learning_rates=None,
                  test_years=None,
                  timestep_col=None, geography_col=None, outcome_col=None,
                  log_dir=None):

    run_template = '{time}_{loc}_{model}_{start_year}_{cov}_{num_inducing}_{lr}'

    town_map = pd.read_csv(os.path.join(data_dir, 'town_tract_map.csv'), dtype=str)
    group_map = gpd.read_file(os.path.join(data_dir, 'tract_group_map'), dtype=str)

    # test y always comes from quarterly tract
    y_timesteps_per_year = 4
    file_name = f'clean_quarter_tract'
    data_path = os.path.join(data_dir, file_name)

    x_idx_cols = [geography_col, 'lat', 'lon', timestep_col,
                  'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc',
                  'svi_pctile',
                  'neighbor_t', 'self_t-1']
    y_idx_cols = [geography_col, timestep_col, outcome_col]
    features_only = ['lat', 'lon', timestep_col,
                     'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc',
                     'svi_pctile',
                     'neighbor_t', 'self_t-1']

    data_gdf = gpd.read_file(data_path)

    last_train_year = 2018

    test_y = data_gdf[(data_gdf['year'] > last_train_year) &
                      (data_gdf['year'] <= last_train_year + test_years)][y_idx_cols]
    starting_y_timestep = int(test_y[timestep_col].min())

    sorted_y_timesteps = test_y[timestep_col].unique()
    sorted_y_timesteps.sort()

    x_timesteps_per_year = {'quarter': 4, 'semi': 2, 'annual': 1}[time]

    file_name = f'clean_{time}_{loc}'
    data_path = os.path.join(data_dir, file_name)

    data_gdf = gpd.read_file(data_path)

    test_x = data_gdf[(data_gdf['year'] > last_train_year) &
                      (data_gdf['year'] <= last_train_year + test_years)][x_idx_cols]

    starting_x_timestep = int(test_x[timestep_col].min())

    test_timesteps_per_year = max(y_timesteps_per_year, x_timesteps_per_year)
    test_timesteps = test_timesteps_per_year * test_years

    x_repeats = int(test_timesteps_per_year / x_timesteps_per_year)
    y_repeats = int(test_timesteps_per_year / y_timesteps_per_year)

    sorted_x_timesteps = test_x[timestep_col].unique()
    sorted_x_timesteps.sort()

    x_timesteps = [timestep for timestep in sorted_x_timesteps for _ in range(x_repeats)]

    y_timesteps = [timestep for timestep in sorted_y_timesteps for _ in range(y_repeats)]

    best_elbo = -np.inf
    for lr in learning_rates:

        this_run = run_template.format(time=time, loc=loc,
                                       model=model, start_year=start_year,
                                       cov=cov,
                                       num_inducing=num_inducing, lr=lr)
        try:
            with open(os.path.join(log_dir, this_run, 'model.mod'), 'rb') as f:
                predictor = pickle.load(f)
            with open(os.path.join(log_dir, this_run, 'stats.csv'), 'rb') as f:
                stats = pd.read_csv(f)
                elbo = stats.iloc[-1, :][['elbo']].values[0]
        except(FileNotFoundError):
            print(os.path.join(log_dir,this_run))
            continue

        if elbo > best_elbo:
            best_elbo = elbo
            best_predictor = copy.deepcopy(predictor)

    xtops = []
    for year in range(test_years):
        xtop_year = []
        max_timesteps = max(x_timesteps_per_year, y_timesteps_per_year)
        for x_time, y_time in zip(x_timesteps[year * max_timesteps:(year + 1) * max_timesteps],
                                  y_timesteps[year * max_timesteps:(year + 1) * max_timesteps]):
            test_x_time = test_x[test_x[timestep_col] == x_time]
            test_y_time = test_y[test_y[timestep_col] == y_time]
            _, _, _, fmean, fvar, gmean, gvar, _, _ = best_predictor.build_predict(test_x_time.loc[:, features_only].values)
            g_cond = tf.math.softplus(fmean * normcdf(gmean)).numpy()
            pred_df = pd.Series(g_cond.squeeze(), index=test_x_time[geography_col])

            if loc == 'town':
                merged_to_map = town_map.merge(pred_df.rename(outcome_col), right_index=True, left_on='parent_town',
                                               how='right')
                averaged_over_duplicates = merged_to_map.groupby('child_tracts').mean()[outcome_col]
                y_index = test_y[geography_col].unique()
                pred_df = pd.Series(index=y_index, dtype='float64')
                pred_df.update(averaged_over_duplicates)
                # not all tracts get mapped from towns
                pred_df = pred_df.fillna(0)
            elif loc == 'group':
                merged_to_map = group_map.merge(pred_df.rename(outcome_col), right_index=True, left_on='grouping',
                                                how='right')
                averaged_over_duplicates = merged_to_map.groupby('geoid').mean()[outcome_col]
                y_index = test_y[geography_col].unique()
                pred_df = pd.Series(index=y_index, dtype='float64')
                pred_df.update(averaged_over_duplicates)

            y_index = test_y_time[geography_col]
            sampled_size = int(0.95*len(y_index))
            assert(sampled_size==1539)
            sampled_xtops = []
            for _ in range(100):
                sampled_index = np.random.choice(y_index, sampled_size,replace=False)
                sampled_test = test_y_time.set_index(geography_col).loc[sampled_index, outcome_col]
                sampled_pred = pred_df.loc[sampled_index]
                result = fixed_top_X(sampled_test, sampled_pred, 100)[-1]
                sampled_xtops.append(result)

            xtop_ptiles = np.percentile(sampled_xtops, [2.5, 50, 97.5])
            xtop_year.append(xtop_ptiles)
        annual_avg = np.array(xtop_year).mean(axis=0)
        xtops.append(annual_avg)

    final_results = pd.DataFrame(xtops)
    final_results.to_csv(os.path.join(log_dir, f'unc_results_{time}_{loc}_{model}_{start_year}_{cov}_{num_inducing}.csv'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help="Path to opioid data file")
    parser.add_argument('--time', type=str)
    parser.add_argument('--loc', type=str)
    parser.add_argument('--start_year', type=int)
    parser.add_argument('--cov', type=str)
    parser.add_argument('--test_years', type=int, help='Number of years to test')
    parser.add_argument('--timestep_col', type=str, default='timestep', help='Name of column containg time index')
    parser.add_argument('--geography_col', type=str, default='geoid', help='Name of column containg geography index')
    parser.add_argument('--outcome_col', type=str, default='deaths', help='Name of column containg geography index')
    parser.add_argument('--model', type=str, default='normal', choices=['normal', 'poisson'])
    parser.add_argument('--num_inducing', type=int, required=True, default=200,
                        help="Number of inducing points to use")
    parser.add_argument('--learning_rates', type=float, nargs="+",
                        help="Adam LR", default=0.005)
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save results in')


    args = parser.parse_args()

    check_results(**vars(args))





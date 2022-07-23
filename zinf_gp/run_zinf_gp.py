"""Run a zero-inflated GP on opioid data"""
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

from math import radians, cos, sin, asin, sqrt
from onoffgpf import OnOffSVGP, OnOffLikelihood

import pickle

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    https://stackoverflow.com/a/4913653/1748679
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def top_X(y_true, y_pred, X=10):
    top_X_predicted = y_pred.sort_values(ascending=False)[:X]
    top_X_true = y_true.sort_values(ascending=False)[:X]

    undisputed_top_predicted = top_X_predicted[top_X_predicted > top_X_predicted.min()]
    num_tied_spots = X - len(undisputed_top_predicted)
    undisputed_top_true = top_X_true[top_X_true > top_X_true.min()]
    num_true_ties = X - len(undisputed_top_true)

    tied_top_predicted = top_X_predicted[top_X_predicted == top_X_predicted.min()]
    tied_top_true = top_X_true[top_X_true == top_X_true.min()]

    error_in_top_true_ties = np.abs(tied_top_true - y_pred[tied_top_true.index]).sort_values(ascending=True)
    error_in_top_pred_ties = np.abs(y_true[tied_top_predicted.index] - tied_top_predicted).sort_values(ascending=True)
    top_true_tied_geoids = error_in_top_true_ties[:num_true_ties].index
    top_pred_tied_geoids = error_in_top_pred_ties[:num_tied_spots].index

    best_possible_top_true_geoids = pd.Index.union(undisputed_top_true.index, top_true_tied_geoids)
    best_possible_top_pred_geoids = pd.Index.union(undisputed_top_predicted.index, top_pred_tied_geoids)

    # True values of GEOIDS with highest actual deaths. If ties, finds tied locations that match preds best
    best_possible_true = y_true[best_possible_top_true_geoids]
    best_possible_pred = y_true[best_possible_top_pred_geoids]

    assert (len(best_possible_true) == X)
    assert (len(best_possible_pred) == X)

    best_possible_absolute = np.abs(best_possible_true.sum() - best_possible_pred.sum())
    best_possible_ratio = np.abs(best_possible_pred).sum() / np.abs(best_possible_true).sum()

    bootstrapped_tied_indices = np.random.choice(tied_top_predicted.index, (1000, num_tied_spots))
    bootstrapped_all_indices = [pd.Index.union(undisputed_top_predicted.index,
                                               bootstrap_index) for bootstrap_index in bootstrapped_tied_indices]

    bootstrapped_absolute = np.mean([np.abs(top_X_true.sum() - y_true[indices].sum())
                                     for indices in bootstrapped_all_indices])
    bootstrapped_ratio = np.mean([np.abs(y_true[indices]).sum() / np.abs(top_X_true).sum()
                                  for indices in bootstrapped_all_indices])

    return best_possible_absolute, best_possible_ratio, bootstrapped_absolute, bootstrapped_ratio


def run_model(time=None, data_dir=None, auto_kernel=False, inducing_points=None, iterations=None,
              out_dir):

    result_dir = os.path.join(data_dir, 'results')
    mass_shapefile = os.path.join(data_dir, 'shapefiles', 'MA_2021')

    svi_file = os.path.join(result_dir, 'svi_month')
    svi_gdf = gpd.read_file(svi_file)
    # Call it "grid_squar" because geopandas only supports len 10 columns
    svi_gdf = svi_gdf.rename(columns={'INTPTLAT': 'lat', 'INTPTLON': 'lon', 'GEOID': 'grid_squar'})
    # Make lat and lon floats
    svi_gdf.loc[:, 'lat'] = svi_gdf.lat.astype(float)
    svi_gdf.loc[:, 'lon'] = svi_gdf.lon.astype(float)
    deaths_gdf = svi_gdf

    # Used when we just need the unique tracts and their locations
    just_grid = deaths_gdf.loc[
        (deaths_gdf['year'] == 2000) & (deaths_gdf['month'] == 1), ['grid_squar', 'geometry', 'lat', 'lon']]

    # Calculate each squares neighbors
    neighbors = {}
    for _, row in just_grid.iterrows():
        just_grid.loc[:, 'haversine'] = just_grid.apply(lambda x: haversine(row['lon'], row['lat'],
                                                                            x['lon'], x['lat']),
                                                        axis=1)
        matching_neighbors = just_grid[just_grid['haversine'] < 8]['grid_squar'].values
        neighbors[row['grid_squar']] = matching_neighbors

    tracts = deaths_gdf['grid_squar'].unique()
    min_year = deaths_gdf.year.min()
    max_year = deaths_gdf.year.max()
    deaths_gdf = deaths_gdf.set_index(['grid_squar', 'year', 'month']).sort_index()

    month_since_2000 = 0
    season_since_2000 = 0
    qtr_since_2000 = 0
    year_since_2000 = 0
    for year in range(min_year, max_year + 1):
        for month in range(1, 12 + 1):

            if month in [1, 2, 3, 4, 5, 6]:
                season = 'jan-jun'
            else:
                season = 'jul-dec'

            if month <= 3:
                qtr = 1
            elif month <= 6:
                qtr = 2
            elif month <= 9:
                qtr = 3
            else:
                qtr = 4

            deaths_gdf.loc[idx[:, year, month], 'month_since_2000'] = month_since_2000
            deaths_gdf.loc[idx[:, year, month], 'season'] = season
            deaths_gdf.loc[idx[:, year, month], 'season_since_2000'] = season_since_2000
            deaths_gdf.loc[idx[:, year, month], 'qtr'] = qtr
            deaths_gdf.loc[idx[:, year, month], 'qtr_since_2000'] = qtr_since_2000
            deaths_gdf.loc[idx[:, year, month], 'year_since_2000'] = year_since_2000

            month_since_2000 += 1

            if month in [6, 12]:
                season_since_2000 += 1

            if month in [3, 6, 9, 12]:
                qtr_since_2000 += 1

            if month == 12:
                year_since_2000 += 1

    deaths_gdf = deaths_gdf.reset_index()
    tracts = deaths_gdf['grid_squar'].unique()
    min_year = deaths_gdf.year.min()
    max_year = deaths_gdf.year.max()
    deaths_gdf = deaths_gdf.set_index(['grid_squar', 'year', 'month']).sort_index()
    deaths_gdf.loc[idx[:, :, :], 'last_timestep'] = deaths_gdf.loc[idx[:, :, :], 'deaths'].shift(1, )
    deaths_gdf.loc[idx[:, :, :], 'last_year'] = deaths_gdf.loc[idx[:, :, :], 'deaths'].shift(12)
    deaths_gdf.loc[idx[:, :, :], 'delta_deaths'] = deaths_gdf.loc[idx[:, :, :], 'deaths'] - deaths_gdf.loc[
        idx[:, :, :], 'last_timestep']
    for tract in tracts:
        deaths_gdf.loc[idx[tract, :, :], 'neighbors_last_timestep'] = \
            deaths_gdf.loc[idx[neighbors[tract], :, :], 'last_timestep'].groupby(level=['year', 'month']).mean().shift(
                1).values
        deaths_gdf.loc[idx[tract, :, :], 'neighbors_last_year'] = \
            deaths_gdf.loc[idx[neighbors[tract], :, :], 'last_year'].groupby(level=['year', 'month']).mean().shift(
                12).values

    deaths_gdf = deaths_gdf.reset_index()

    if time=='qtr':
        timestep_col = 'qtr_since_2000'
        deaths_gdf = deaths_gdf.groupby(['grid_squar', 'year', timestep_col]).sum(min_count=3)[
            ['deaths', 'delta_deaths', 'last_timestep', 'last_year', 'neighbors_last_timestep', 'neighbors_last_year']]
        deaths_gdf_meta = deaths_gdf.groupby(['grid_squar', 'year', timestep_col]).mean()[
            ['theme_3_pc', 'lon', 'theme_2_pc', 'lat', 'svi_pctile', 'theme_1_pc', 'theme_4_pc']]
        deaths_gdf = deaths_gdf.merge(deaths_gdf_meta, left_index=True, right_index=True)
        deaths_gdf.loc[idx[:, :, :], 'last_timestep'] = deaths_gdf.loc[idx[:, :, :], 'deaths'].shift(1, )
        deaths_gdf.loc[idx[:, :, :], 'last_year'] = deaths_gdf.loc[idx[:, :, :], 'deaths'].shift(2)
        deaths_gdf.loc[idx[:, :, :], 'delta_deaths'] = deaths_gdf.loc[idx[:, :, :], 'deaths'] - \
                                                       deaths_gdf.loc[idx[:, :], 'last_timestep']
        for tract in tracts:
            deaths_gdf.loc[idx[tract, :, :], 'neighbors_last_timestep'] = \
                deaths_gdf.loc[idx[neighbors[tract], :, :], 'last_timestep'].groupby(
                    level=['season_since_2000', 'year']).mean().shift(1).values
            deaths_gdf.loc[idx[tract, :, :], 'neighbors_last_year'] = \
                deaths_gdf.loc[idx[neighbors[tract], :, :], 'last_year'].groupby(
                    level=['season_since_2000', 'year']).mean().shift(4).values
    elif time=='biannual':
        timestep_col = 'season_since_2000'
        deaths_gdf = deaths_gdf.groupby(['grid_squar', 'year', timestep_col]).sum(min_count=6)[
            ['deaths', 'delta_deaths', 'last_timestep', 'last_year', 'neighbors_last_timestep', 'neighbors_last_year']]
        deaths_gdf_meta = deaths_gdf.groupby(['grid_squar', 'year', timestep_col]).mean()[
            ['theme_3_pc', 'lon', 'theme_2_pc', 'lat', 'svi_pctile', 'theme_1_pc', 'theme_4_pc']]
        deaths_gdf = deaths_gdf.merge(deaths_gdf_meta, left_index=True, right_index=True)
        deaths_gdf.loc[idx[:, :, :], 'last_timestep'] = deaths_gdf.loc[idx[:, :, :], 'deaths'].shift(1, )
        deaths_gdf.loc[idx[:, :, :], 'last_year'] = deaths_gdf.loc[idx[:, :, :], 'deaths'].shift(2)
        deaths_gdf.loc[idx[:, :, :], 'delta_deaths'] = deaths_gdf.loc[idx[:, :, :], 'deaths'] - \
                                                              deaths_gdf.loc[idx[:, :], 'last_timestep']
        for tract in tracts:
            deaths_gdf.loc[idx[tract, :, :], 'neighbors_last_timestep'] = \
                deaths_gdf.loc[idx[neighbors[tract], :, :], 'last_timestep'].groupby(
                    level=[timestep_col, 'year']).mean().shift(1).values
            deaths_gdf.loc[idx[tract, :, :], 'neighbors_last_year'] = \
                deaths_gdf.loc[idx[neighbors[tract], :, :], 'last_year'].groupby(
                    level=[timestep_col, 'year']).mean().shift(2).values

    elif time=='annual':
        timestep_col = 'year_since_2000'
        deaths_gdf = deaths_gdf.groupby(['grid_squar', 'year', timestep_col]).sum(min_count=6)[
            ['deaths', 'delta_deaths', 'last_timestep', 'last_year', 'neighbors_last_timestep', 'neighbors_last_year']]
        deaths_gdf_meta = deaths_gdf.groupby(['grid_squar', 'year', timestep_col]).mean()[
            ['theme_3_pc', 'lon', 'theme_2_pc', 'lat', 'svi_pctile', 'theme_1_pc', 'theme_4_pc']]
        deaths_gdf = deaths_gdf.merge(deaths_gdf_meta, left_index=True, right_index=True)
        deaths_gdf.loc[idx[:, :, :], 'last_timestep'] = deaths_gdf.loc[idx[:, :, :], 'deaths'].shift(1, )
        deaths_gdf.loc[idx[:, :, :], 'last_year'] = deaths_gdf.loc[idx[:, :, :], 'deaths'].shift(2)
        deaths_gdf.loc[idx[:, :, :], 'delta_deaths'] = deaths_gdf.loc[idx[:, :, :], 'deaths'] - \
                                                       deaths_gdf.loc[idx[:, :], 'last_timestep']
        for tract in tracts:
            deaths_gdf.loc[idx[tract, :, :], 'neighbors_last_timestep'] = \
                deaths_gdf.loc[idx[neighbors[tract], :, :], 'last_timestep'].groupby(
                    level=[timestep_col, 'year']).mean().shift(1).values
            # sameas timestep in this case
            deaths_gdf.loc[idx[tract, :, :], 'neighbors_last_year'] = \
                deaths_gdf.loc[idx[neighbors[tract], :, :], 'last_year'].groupby(
                    level=[timestep_col, 'year']).mean().shift(1).values

    deaths_gdf_with_autoregressive = deaths_gdf.reset_index()

    if auto_kernel:
        features = ['grid_squar', 'lat', 'lon', timestep_col, 'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc',
         'svi_pctile', 'neighbors_last_timestep', 'last_timestep']
    else:
        features = ['grid_squar', 'lat', 'lon', timestep_col, 'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc',
         'svi_pctile']

    train_x_through_2018 = deaths_gdf_with_autoregressive[deaths_gdf_with_autoregressive['year'] <= 2018][features].dropna()
    train_y_through_2018 = deaths_gdf_with_autoregressive.loc[train_x_through_2018.index][
        ['grid_squar', timestep_col, 'deaths']].dropna()
    train_x_through_2019 = deaths_gdf_with_autoregressive[deaths_gdf_with_autoregressive['year'] <= 2019][
        features].dropna()
    train_y_through_2019 = deaths_gdf_with_autoregressive.loc[train_x_through_2019.index][
        ['grid_squar', timestep_col, 'deaths']].dropna()

    x_just_2019 = deaths_gdf_with_autoregressive[deaths_gdf_with_autoregressive['year'] == 2019][
        features]
    y_just_2019 = deaths_gdf_with_autoregressive[deaths_gdf_with_autoregressive['year'] == 2019][
        ['grid_squar', timestep_col, 'deaths']]
    x_just_2020 = deaths_gdf_with_autoregressive[deaths_gdf_with_autoregressive['year'] == 2020][
        features]
    y_just_2020 = deaths_gdf_with_autoregressive[deaths_gdf_with_autoregressive['year'] == 2020][
        ['grid_squar', timestep_col, 'deaths']]

    spatial_kernel = gpflow.kernels.RBF(2, active_dims=[0, 1])
    temporal_kernel = gpflow.kernels.RBF(1, active_dims=[2])
    demo_kernel = gpflow.kernels.RBF(5, active_dims=[3, 4, 5, 6, 7])

    if auto_kernel:
        autoregressive_kernel = gpflow.kernels.RBF(2, active_dims=[8,9])
        gaussian_kernel = spatial_kernel + temporal_kernel + demo_kernel + autoregressive_kernel
    else:
        gaussian_kernel = spatial_kernel + temporal_kernel + demo_kernel

    f_kernel = copy.deepcopy(gaussian_kernel)
    g_kernel = copy.deepcopy(gaussian_kernel)
    likelihood = OnOffLikelihood()

    random = np.random.default_rng(seed=1)

    M = inducing_points

    N = len(train_x_through_2018)
    Z = random.choice(train_x_through_2018[
                          features].values, size=M, replace=False)

    Zf = copy.deepcopy(Z)
    Zg = copy.deepcopy(Z)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x_through_2018.loc[:,
                                                        features],
                                                        train_y_through_2018.loc[:, 'deaths'].values.reshape(-1,
                                                                                                             1))).repeat().shuffle(
        N)

    m = OnOffSVGP(train_x_through_2018.loc[:, features].values,
                  train_y_through_2018.loc[:, 'deaths'].values.reshape(-1, 1)
                  , kernf=f_kernel,
                  kerng=g_kernel
                  , likelihood=OnOffLikelihood()
                  , Zf=Zf,
                  Zg=Zg
                  )

    # fix the model noise term
    m.likelihood.variance = 0.01
    m.likelihood.variance.fixed = False

    m.optimize(maxiter=iterations)  # ,method= tf.train.AdamOptimizer(learning_rate = 0.01)

    pred_2019 = m.predict_onoffgp(x_just_2019.loc[:, features].values)
    pred_2019 = pred_2019[0]

    pred_timesteps = pred_2019.timesteps.unique()

    maes, top_10s, top_50s, top_100s = [], [], [], []
    for timestep in pred_timesteps:
        single_time_pred = pred_2019[pred_2019['timestep']==timestep]
        single_time_true = y_just_2019[y_just_2019['timestep']==timestep]

        maes.append(mean_absolute_error(single_time_true.deaths, single_time_pred))
        top_10s.append(top_X(single_time_true.set_index('grid_squar')['deaths'], single_time_pred, 10))
        top_50s.append(top_X(single_time_true.set_index('grid_squar')['deaths'], single_time_pred, 50))
        top_100s.append(top_X(single_time_true.set_index('grid_squar')['deaths'], single_time_pred, 100))

    model_fname = os.path.join(out_dir, 'model.pkl')
    result_fname = os.path.join(out_dir, 'res,pkl')

    m.savemodel(model_fname)
    with open(result_fname, 'wb') as outfile:
        pickle.dump(result_fname)


    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser

    parser.add_argument('--time', type=str, help="Temporal division of data",
                        choices=['qtr','biannual', 'annual'])
    parser.add_argument('--data_dir', type=str, help="Path to opioid data",
                        default='/cluster/tufts/hugheslab/datasets/NSF_OD/')
    parser.add_argument('--auto_kernel', action='store_true', help="If present, add a kernel with autoregressive features.")
    parser.add_argument('--inducing_points', type=int, required=True,
                        help="Number of inducing points to use")
    parser.add_argument('--iterations', type=int, required=True,
                        help="Number of iterations to run")
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save results in')
    
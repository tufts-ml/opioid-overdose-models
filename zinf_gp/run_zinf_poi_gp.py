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

code_dir = '/cluster/home/kheuto01/code/zero-inflated-gp/'
sys.path.append(code_dir)

from onoffgpf import OnOffSVGPPoiMC, OnOffLikelihood
gpflow.config.default_float()

from math import radians, cos, sin, asin, sqrt

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

def normcdf(x):
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1. - 2.e-3) + 1.e-3


def fixed_top_X(true_qtr_val, pred_qtr_val, X=10):
    top_X_predicted = pred_qtr_val.sort_values(ascending=False)[:X]
    top_X_true = true_qtr_val.sort_values(ascending=False)[:X]

    undisputed_top_predicted = top_X_predicted[top_X_predicted > top_X_predicted.min()]
    num_tied_spots = X - len(undisputed_top_predicted)
    undisputed_top_true = top_X_true[top_X_true > top_X_true.min()]
    num_true_ties = X - len(undisputed_top_true)

    tied_top_predicted = pred_qtr_val[pred_qtr_val == top_X_predicted.min()]
    tied_top_true = true_qtr_val[true_qtr_val == top_X_true.min()]

    error_in_top_true_ties = np.abs(tied_top_true - pred_qtr_val[tied_top_true.index]).sort_values(ascending=True)
    error_in_top_pred_ties = np.abs(true_qtr_val[tied_top_predicted.index] - tied_top_predicted).sort_values(
        ascending=True)
    top_true_tied_geoids = error_in_top_true_ties[:num_true_ties].index
    top_pred_tied_geoids = error_in_top_pred_ties[:num_tied_spots].index

    best_possible_top_true_geoids = pd.Index.union(undisputed_top_true.index, top_true_tied_geoids)
    best_possible_top_pred_geoids = pd.Index.union(undisputed_top_predicted.index, top_pred_tied_geoids)

    # True values of GEOIDS with highest actual deaths. If ties, finds tied locations that match preds best
    best_possible_true = true_qtr_val[best_possible_top_true_geoids]
    best_possible_pred = true_qtr_val[best_possible_top_pred_geoids]

    assert (len(best_possible_true) == X)
    assert (len(best_possible_pred) == X)

    best_possible_absolute = np.abs(best_possible_true.sum() - best_possible_pred.sum())
    best_possible_ratio = np.abs(best_possible_pred).sum() / np.abs(best_possible_true).sum()

    bootstrapped_tied_indices = np.random.choice(tied_top_predicted.index, (1000, num_tied_spots))
    bootstrapped_all_indices = [pd.Index.union(undisputed_top_predicted.index,
                                               bootstrap_index) for bootstrap_index in bootstrapped_tied_indices]

    bootstrapped_absolute = np.mean([np.abs(top_X_true.sum() - true_qtr_val[indices].sum())
                                     for indices in bootstrapped_all_indices])
    bootstrapped_ratio = np.mean([np.abs(true_qtr_val[indices]).sum() / np.abs(top_X_true).sum()
                                  for indices in bootstrapped_all_indices])

    return best_possible_absolute, best_possible_ratio, bootstrapped_absolute, bootstrapped_ratio


def run_adam(model, iterations, out_dir, death_df, learning_rate=0.005):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    stat_logs = pd.DataFrame()
    stat_path = os.path.join(out_dir, 'stats.csv')
    model_path = os.path.join(out_dir, 'model.mod')
    training_loss = model.training_loss_closure(compile=True)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)#gpflow.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)

            maes = []
            xtop = []
            for quarter in range(1, 4 + 1):
                test_x = death_df[(death_df['year'] == 2019) & (
                            death_df['quarter'] == quarter)][
                    ['grid_squar', 'lat', 'lon', 'timestep', 'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc',
                     'svi_pctile', 'neighbor_t-1', 'self_t-1']]
                test_y = death_df[(death_df['year'] == 2019) & (
                            death_df['quarter'] == quarter)][['grid_squar', 'timestep', 'deaths']]
                _, _, _, fmean, fvar, gmean, gvar, _, _ = model.build_predict(test_x.loc[:,
                                                                          ['lat', 'lon', 'timestep', 'theme_1_pc',
                                                                           'theme_2_pc', 'theme_3_pc', 'theme_4_pc',
                                                                           'svi_pctile', 'neighbor_t-1',
                                                                           'self_t-1']].values)
                sg_2019 = tf.math.softplus(fmean * normcdf(gmean) + 2).numpy()
                pred_2019_df = pd.Series(sg_2019.squeeze(), index=test_y.grid_squar)

                maes.append(mean_absolute_error(test_y.deaths, pred_2019_df))
                xtop.append(fixed_top_X(test_y.set_index('grid_squar')['deaths'], pred_2019_df, 100))

                stat_logs = stat_logs.append({'iter': step,'elbo':elbo, 'mae': np.mean(maes), 'bpr_100': np.mean(xtop)},
                                             ignore_index=True)

                stat_logs.to_csv(stat_path)


    return logf


def run_model(time=None, data_dir=None, kernel=None, auto_kernel=False, inducing_points=None, iterations=None,
              out_dir=None, samples=None, learning_rate=None):

    result_dir = os.path.join(data_dir, 'results')
    mass_shapefile = os.path.join(data_dir, 'shapefiles', 'MA_2021')

    svi_file = os.path.join(result_dir, 'svi_qtr')
    svi_gdf = gpd.read_file(svi_file)
    svi_gdf = svi_gdf.rename(columns={'INTPTLAT': 'lat', 'INTPTLON': 'lon', 'GEOID': 'grid_squar'})
    # Make lat and lon floats
    svi_gdf.loc[:, 'lat'] = svi_gdf.lat.astype(float)
    svi_gdf.loc[:, 'lon'] = svi_gdf.lon.astype(float)
    deaths_gdf = svi_gdf
    just_grid = deaths_gdf.loc[
        (deaths_gdf['year'] == 2000) & (deaths_gdf['quarter'] == 4), ['grid_squar', 'geometry', 'lat', 'lon']]

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
    deaths_gdf = deaths_gdf.set_index(['grid_squar', 'year', 'quarter']).sort_index()
    deaths_gdf.loc[idx[:, :, :], 'self_t-1'] = deaths_gdf.loc[idx[:, :, :], 'deaths'].shift(1, fill_value=0)
    for tract in tracts:
        deaths_gdf.loc[idx[tract, :, :], 'neighbor_t-1'] = \
            deaths_gdf.loc[idx[neighbors[tract], :, :], 'self_t-1'].groupby(level=['year', 'quarter']).mean().shift(1,
                                                                                                                    fill_value=0).values

    timestep = 0

    for year in range(min_year, max_year + 1):
        for quarter in range(1, 5):
            deaths_gdf.loc[idx[:, year, quarter], 'timestep'] = timestep
            timestep += 1

    deaths_gdf_with_autoregressive = deaths_gdf.reset_index()

    train_x_through_2018 = deaths_gdf_with_autoregressive[deaths_gdf_with_autoregressive['year'] <= 2018][
        ['grid_squar', 'lat', 'lon', 'timestep', 'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc', 'svi_pctile',
         'neighbor_t-1', 'self_t-1']]
    train_y_through_2018 = deaths_gdf_with_autoregressive[deaths_gdf_with_autoregressive['year'] <= 2018][
        ['grid_squar', 'timestep', 'deaths']]
    train_x_through_2019 = deaths_gdf_with_autoregressive[deaths_gdf_with_autoregressive['year'] <= 2019][
        ['grid_squar', 'lat', 'lon', 'timestep', 'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc', 'svi_pctile',
         'neighbor_t-1', 'self_t-1']]
    train_y_through_2019 = deaths_gdf_with_autoregressive[deaths_gdf_with_autoregressive['year'] <= 2019][
        ['grid_squar', 'timestep', 'deaths']]

    x_just_2019 = deaths_gdf_with_autoregressive[deaths_gdf_with_autoregressive['year'] == 2019][
        ['grid_squar', 'lat', 'lon', 'timestep', 'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc', 'svi_pctile',
         'neighbor_t-1', 'self_t-1']]
    y_just_2019 = deaths_gdf_with_autoregressive[deaths_gdf_with_autoregressive['year'] == 2019][
        ['grid_squar', 'timestep', 'deaths']]
    x_just_2020 = deaths_gdf_with_autoregressive[deaths_gdf_with_autoregressive['year'] == 2020][
        ['grid_squar', 'lat', 'lon', 'timestep', 'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc', 'svi_pctile',
         'neighbor_t-1', 'self_t-1']]
    y_just_2020 = deaths_gdf_with_autoregressive[deaths_gdf_with_autoregressive['year'] == 2020][
        ['grid_squar', 'timestep', 'deaths']]

    x_just_2019q1 = deaths_gdf_with_autoregressive[
        (deaths_gdf_with_autoregressive['year'] == 2019) & (deaths_gdf_with_autoregressive['quarter'] == 1)][
        ['grid_squar', 'lat', 'lon', 'timestep', 'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc', 'svi_pctile',
         'neighbor_t-1', 'self_t-1']]
    y_just_2019q1 = deaths_gdf_with_autoregressive[
        (deaths_gdf_with_autoregressive['year'] == 2019) & (deaths_gdf_with_autoregressive['quarter'] == 1)][
        ['grid_squar', 'timestep', 'deaths']]

    spatial_kernel = gpflow.kernels.RBF(2, active_dims=[0, 1])
    temporal_kernel = gpflow.kernels.RBF(1, active_dims=[2])

    if kernel == 'st_only':
        gaussian_kernel = spatial_kernel + temporal_kernel
    elif kernel == 'svi_only':
        demo_kernel = gpflow.kernels.RBF(1, active_dims=[7])
        gaussian_kernel = spatial_kernel + temporal_kernel + demo_kernel
    elif kernel == 'svi_full':
        demo_kernel = gpflow.kernels.RBF(5, active_dims=[3, 4, 5, 6, 7])
        gaussian_kernel = spatial_kernel + temporal_kernel + demo_kernel

    if auto_kernel:
        autoregressive_kernel = gpflow.kernels.RBF(2, active_dims=[8, 9])
        gaussian_kernel = gaussian_kernel + autoregressive_kernel

    f_kernel = copy.deepcopy(gaussian_kernel)
    g_kernel = copy.deepcopy(gaussian_kernel)
    likelihood = OnOffLikelihood()
    random = np.random.default_rng(seed=1)

    M = inducing_points
    N = len(train_x_through_2018)
    Z = random.choice(train_x_through_2018[
                          ['lat', 'lon', 'timestep', 'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc',
                           'svi_pctile', 'neighbor_t-1', 'self_t-1']].values, size=M, replace=False)

    Zf = copy.deepcopy(Z)
    Zg = copy.deepcopy(Z)



    m = OnOffSVGPPoiMC(train_x_through_2018.loc[:, ['lat', 'lon', 'timestep', 'theme_1_pc', 'theme_2_pc',
                                                    'theme_3_pc', 'theme_4_pc', 'svi_pctile',
                                                    'neighbor_t-1', 'self_t-1']].values,
                       train_y_through_2018.loc[:, 'deaths'].values.reshape(-1, 1)
                       , kernf=f_kernel,
                       kerng=g_kernel
                       , likelihood=OnOffLikelihood()
                       , Zf=Zf,
                       Zg=Zg,
                       samples=samples
                       )

    logf = run_adam(m, 2000,out_dir, deaths_gdf_with_autoregressive, learning_rate=learning_rate)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--time', type=str, help="Temporal division of data",
                        choices=['qtr','biannual', 'annual'], default='qtr')
    parser.add_argument('--data_dir', type=str, help="Path to opioid data",
                        default='/cluster/tufts/hugheslab/datasets/NSF_OD/')
    parser.add_argument('--kernel', type=str, help="How to make kernels",
                        choices=['st_only', 'svi_only', 'svi_full'],)
    parser.add_argument('--auto_kernel', action='store_true', help="If present, add a kernel with autoregressive features.")
    parser.add_argument('--inducing_points', type=int, required=True, default=200,
                        help="Number of inducing points to use")
    parser.add_argument('--samples', type=int, default=10,
                        help="Number of inducing points to use")
    parser.add_argument('--iterations', type=int, required=True,
                        help="Number of iterations to run")
    parser.add_argument('--learning_rate', type=float,
                        help="Adam LR", default=0.005)
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save results in')


    args = parser.parse_args()
    
    run_model(**vars(args))    

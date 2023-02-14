import copy
import sys

import geopandas as gpd
import numpy as np

import gpflow

from metrics import fixed_top_X
from model_runner import run_adam

code_dir = '/cluster/home/kheuto01/code/zero-inflated-gp/'
sys.path.append(code_dir)

from onoffgpf import OnOffSVGP, OnOffSVGPPoiMC, OnOffLikelihood
gpflow.config.default_float()



def run_model(data_path=None, last_train_year=None, test_years=None,
              timesteps_per_year=None,
              timestep_col=None, geography_col=None, outcome_col=None,
              use_auto=None, use_svi=None,
              likelihood=None, seed=None,
              inducing_points=None, samples=None, iterations=None,
              learning_rate=None,
              out_dir=None):

    x_idx_cols = [geography_col, 'lat','lon', timestep_col,
                  'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc',
                  'svi_pctile',
                  'neighbor_t-1', 'self_t-1']
    y_idx_cols = [geography_col, timestep_col, outcome_col]
    features_only = ['lat','lon', timestep_col,
                     'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc',
                     'svi_pctile',
                     'neighbor_t-1', 'self_t-1']

    data_gdf = gpd.read_file(data_path)

    train_x = data_gdf[data_gdf['year'] <= last_train_year][x_idx_cols]
    train_y = data_gdf[data_gdf['year'] <= last_train_year][y_idx_cols]
    test_x = data_gdf[(data_gdf['year'] > last_train_year) &
                      (data_gdf['year'] <= last_train_year+test_years)][x_idx_cols]
    test_y = data_gdf[(data_gdf['year'] > last_train_year) &
                      (data_gdf['year'] <= last_train_year+test_years)][y_idx_cols]

    # just lat and lon
    spatial_kernel = gpflow.kernels.RBF(2, active_dims=[0, 1])
    # timestep col is both an index and a feature
    temporal_kernel = gpflow.kernels.RBF(1, active_dims=[2])
    # all 5 features of svi
    svi_kernel = gpflow.kernels.RBF(5, active_dims=[3, 4, 5, 6, 7])
    # just the autoregressive features
    auto_kernel = gpflow.kernels.RBF(2, active_dims=[8, 9])

    full_kernel = spatial_kernel + temporal_kernel
    if use_auto:
        full_kernel += auto_kernel
    if use_svi:
        full_kernel += svi_kernel

    f_kernel = copy.deepcopy(full_kernel)
    g_kernel = copy.deepcopy(full_kernel)

    latent_likelihood = OnOffLikelihood()

    # initialize inducing points to random data points
    random = np.random.default_rng(seed=seed)
    Z = random.choice(train_x[features_only].values,
                      size=inducing_points, replace=False)

    # Make inducing points for both f and g
    Zf = copy.deepcopy(Z)
    Zg = copy.deepcopy(Z)

    if likelihood == 'normal':
        model = OnOffSVGP(train_x.loc[:, features_only].values,
                          train_y.loc[:, outcome_col].values.reshape(-1, 1),
                          kernf=f_kernel, kerng=g_kernel,
                          likelihood=OnOffLikelihood(),
                          Zf=Zf, Zg=Zg)
    elif likelihood == 'poisson':
        model = OnOffSVGPPoiMC(train_x.loc[:, features_only].values,
                               train_y.loc[:, outcome_col].values.reshape(-1, 1),
                               kernf=f_kernel, kerng=g_kernel,
                               likelihood=OnOffLikelihood(),
                               Zf=Zf, Zg=Zg, samples=samples)
    else:
        print('never ahppens')
        break

    logs = run_adam(model, iterations, learning_rate, out_dir, test_x, test_y,
                    timesteps_per_year, test_years, timestep_col, features_only)

    print(logs)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--time', type=str, help="Temporal division of data",
                        choices=['qtr' ,'biannual', 'annual'], default='qtr')
    parser.add_argument('--data_path', type=str, help="Path to opioid data file")
    parser.add_argument('--last_train_timestep', type=int, help='Value of last timestep used in training')
    parser.add_argument('--timesteps_per_year', type=int, help='Number of timesteps per year')
    parser.add_argument('--test_years', type=int, help='Number of years to test')
    parser.add_argument('--timestep_col', type=str, default='timestep', help='Name of column containg time index')
    parser.add_argument('--geograph_col', type=str, default='geoid', help='Name of column containg geography index')
    parser.add_argument('--kernel', type=str, help="How to make kernels",
                        choicetest_xs=['st_only', 'svi_only', 'svi_full'] ,)
    parser.add_argument('--auto_kernel', action='store_true', help="If present, add a kernel with autoregressive features.")
    parser.add_argument('--likelihood', type=str, default='normal', choices=['normal', 'poisson'])
    parser.add_argument('--inducing_points', type=int, required=True, default=200,
                        help="Number of inducing points to use")
    parser.add_argument('--samples', type=int, default=10,
                        help="Number of inducing points to use")
    parser.add_argument('--iterations', type=int, required=True,
                        help="Number of iterations to run")
    parser.add_argument('--seed', type=int, default=1,
                        help="seed to use for inducing points")
    parser.add_argument('--learning_rate', type=float,
                        help="Adam LR", default=0.005)
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save results in')


    args = parser.parse_args()

    run_model(**vars(args))

import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import IndexSlice as idx

import sklearn
from sklearn.linear_model import LinearRegression

from metrics import fast_bpr


def all_zeroes_model(multiindexed_gdf, first_pred_time, last_pred_time, num_locations, timestep_col='timestep',
                     location_col='geoid', outcome_col='deaths',
                     removed_locations=250, bpr_uncertainty_samples=50, seed=360):

    rng = np.random.default_rng(seed=seed)
    num_sampled = num_locations - removed_locations
    results_over_time = []

    for timestep in range(first_pred_time, last_pred_time+1):
        evaluation_deaths = multiindexed_gdf.loc[idx[:, timestep], :]
        evaluation_deaths = evaluation_deaths.drop(columns=timestep_col).reset_index().set_index(location_col)[outcome_col]

        results_over_samples = []

        for _ in range(bpr_uncertainty_samples):
            sampled_indicies = rng.choice(range(num_locations), size=num_sampled, replace=False)
            results_over_samples.append(fast_bpr(evaluation_deaths[sampled_indicies], evaluation_deaths[sampled_indicies]*0))

        results_over_time.append(results_over_samples)

    return results_over_time


def last_time_model(multiindexed_gdf, first_pred_time, last_pred_time, num_locations,
                     pred_lag,
                     timestep_col='timestep',
                     location_col='geoid', outcome_col='deaths',
                     removed_locations=250, bpr_uncertainty_samples=50, seed=360):
    """pred_lag is how many timesteps between the  time used for prediction and evaluation"""

    rng = np.random.default_rng(seed=seed)
    num_sampled = num_locations - removed_locations
    results_over_time = []

    for timestep in range(first_pred_time, last_pred_time+1):
        evaluation_deaths = multiindexed_gdf.loc[idx[:, timestep], :]
        evaluation_deaths = evaluation_deaths.drop(columns=timestep_col).reset_index().set_index(location_col)[outcome_col]

        predicted_deaths = multiindexed_gdf.loc[idx[:, timestep-pred_lag], :]
        predicted_deaths = predicted_deaths.drop(columns=timestep_col).reset_index().set_index(location_col)[
            outcome_col]

        results_over_samples = []

        for _ in range(bpr_uncertainty_samples):
            sampled_indicies = rng.choice(range(num_locations), size=num_sampled, replace=False)
            results_over_samples.append(fast_bpr(evaluation_deaths[sampled_indicies], predicted_deaths[sampled_indicies]))

        results_over_time.append(results_over_samples)

    return results_over_time

def historical_average_model(multiindexed_gdf, first_pred_time, last_pred_time, num_locations,
                     pred_lag, window_length,
                     timestep_col='timestep',
                     location_col='geoid', outcome_col='deaths',
                     removed_locations=250, bpr_uncertainty_samples=50, seed=360):
    """pred_lag is how many timesteps between the  time used for prediction and evaluation
    window_length is how many years to include"""

    rng = np.random.default_rng(seed=seed)
    num_sampled = num_locations - removed_locations
    results_over_time = []

    for timestep in range(first_pred_time, last_pred_time+1):
        evaluation_deaths = multiindexed_gdf.loc[idx[:, timestep], :]
        evaluation_deaths = evaluation_deaths.drop(columns=timestep_col).reset_index().set_index(location_col)[outcome_col]

        predicted_deaths =  multiindexed_gdf[(multiindexed_gdf[timestep_col]<=timestep-pred_lag) &
                                             (multiindexed_gdf[timestep_col]>timestep-pred_lag-window_length)]
        predicted_deaths = predicted_deaths.groupby(level='geoid')['deaths'].mean()

        results_over_samples = []

        for _ in range(bpr_uncertainty_samples):
            sampled_indicies = rng.choice(range(num_locations), size=num_sampled, replace=False)
            results_over_samples.append(fast_bpr(evaluation_deaths[sampled_indicies], predicted_deaths[sampled_indicies]))

        results_over_time.append(results_over_samples)

    return results_over_time


def scikit_model(multiindexed_gdf, x_BSF, y_BS, test_x_BSF, model,
                 timestep_col='timestep',
                 location_col='geoid', outcome_col='deaths',
                 removed_locations=250, seed=360, bpr_uncertainty_samples=50):

    # B = timesteps of data, # S = locations, # F = features
    B, S, F = x_BSF.shape

    # reshape data into 2D for scikit learn models. 1 row = 1 location at 1 time
    x_long = tf.reshape(x_BSF, ((B * S), F))
    y_long = tf.reshape(y_BS, ((B * S), 1))

    reg = model.fit(x_long, tf.squeeze(y_long))

    # sloppy notation here, it's not the same B
    num_test_times = test_x_BSF.shape[0]

    rng = np.random.default_rng(seed=seed)
    num_sampled = S - removed_locations
    results_over_time = []

    for timestep in range(num_test_times):
        evaluation_deaths = multiindexed_gdf.loc[idx[:, timestep], :]
        evaluation_deaths = evaluation_deaths.drop(columns=timestep_col).reset_index().set_index(location_col)[
            outcome_col]

        prediction = reg.predict(test_x_BSF[0])

        results_over_samples = []

        for _ in range(bpr_uncertainty_samples):
            sampled_indicies = rng.choice(range(S), size=num_sampled, replace=False)

            results_over_samples.append(
                fast_bpr(evaluation_deaths[sampled_indicies],
                         pd.Series(prediction[sampled_indicies],
                                   index=evaluation_deaths[sampled_indicies].index)
                         )
            )

        results_over_time.append(results_over_samples)

    return results_over_time

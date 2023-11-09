import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import IndexSlice as idx
from metrics import fast_bpr
import pickle
import os 
import sys

# import CASTNet files
current_directory = os.path.dirname(os.path.abspath(__file__))
path_to_CASTNet = os.path.abspath(os.path.join(current_directory, '..', 'CASTNet/hughes-CASTNet/'))
sys.path.append(path_to_CASTNet)
import CASTNetWrapper
import hughes_castnet_main

##########

def all_zeroes_model(multiindexed_gdf, first_pred_time, last_pred_time, num_locations, timestep_col='timestep',
                     location_col='geoid', outcome_col='deaths',
                     removed_locations=250, bpr_uncertainty_samples=50, seed=360, bpr_K=100):

    rng = np.random.default_rng(seed=seed)
    num_sampled = num_locations - removed_locations
    results_over_time = []
    predicted_deaths=[]
    denominator_deaths = []

    for timestep in range(first_pred_time, last_pred_time+1):
        evaluation_deaths = multiindexed_gdf.loc[idx[:, timestep], :]
        evaluation_deaths = evaluation_deaths.drop(columns=timestep_col).reset_index().set_index(location_col)[outcome_col]
        predicted_deaths.append(evaluation_deaths*0)
        results_over_samples = []

        for _ in range(bpr_uncertainty_samples):
            sampled_indicies = rng.choice(range(num_locations), size=num_sampled, replace=False)
            denominator_deaths.append(evaluation_deaths[sampled_indicies].sort_values().iloc[-100:].sum())
            results_over_samples.append(fast_bpr(evaluation_deaths[sampled_indicies], evaluation_deaths[sampled_indicies]*0, K=bpr_K))
            #predicted_deaths.append(evaluation_deaths[sampled_indicies]*0)

        results_over_time.append(results_over_samples)

    return results_over_time, predicted_deaths, denominator_deaths


def last_time_model(multiindexed_gdf, first_pred_time, last_pred_time, num_locations,
                     pred_lag,
                     timestep_col='timestep',
                     location_col='geoid', outcome_col='deaths',
                     removed_locations=250, bpr_uncertainty_samples=50, seed=360):
    """pred_lag is how many timesteps between the  time used for prediction and evaluation"""

    rng = np.random.default_rng(seed=seed)
    num_sampled = num_locations - removed_locations
    results_over_time = []
    output_deaths =[] #to store predictions

    for timestep in range(first_pred_time, last_pred_time+1):
        evaluation_deaths = multiindexed_gdf.loc[idx[:, timestep], :]
        evaluation_deaths = evaluation_deaths.drop(columns=timestep_col).reset_index().set_index(location_col)[outcome_col]

        predicted_deaths = multiindexed_gdf.loc[idx[:, timestep-pred_lag], :]
        predicted_deaths = predicted_deaths.drop(columns=timestep_col).reset_index().set_index(location_col)[
            outcome_col]
        
        output_deaths.append(predicted_deaths)

        results_over_samples = []

        for _ in range(bpr_uncertainty_samples):
            sampled_indicies = rng.choice(range(num_locations), size=num_sampled, replace=False)
            results_over_samples.append(fast_bpr(evaluation_deaths[sampled_indicies], predicted_deaths[sampled_indicies]))

        results_over_time.append(results_over_samples)

    return results_over_time, output_deaths

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
    output_deaths = []

    for timestep in range(first_pred_time, last_pred_time+1):
        evaluation_deaths = multiindexed_gdf.loc[idx[:, timestep], :]
        evaluation_deaths = evaluation_deaths.drop(columns=timestep_col).reset_index().set_index(location_col)[outcome_col]

        predicted_deaths =  multiindexed_gdf[(multiindexed_gdf[timestep_col]<=timestep-pred_lag) &
                                             (multiindexed_gdf[timestep_col]>timestep-pred_lag-window_length)]
        predicted_deaths = predicted_deaths.groupby(level='geoid')['deaths'].mean()
        output_deaths.append(predicted_deaths)
        results_over_samples = []

        for _ in range(bpr_uncertainty_samples):
            sampled_indicies = rng.choice(range(num_locations), size=num_sampled, replace=False)
            results_over_samples.append(fast_bpr(evaluation_deaths[sampled_indicies], predicted_deaths[sampled_indicies]))


        results_over_time.append(results_over_samples)

    return results_over_time, output_deaths


def scikit_model(multiindexed_gdf, x_BSF, y_BS, test_x_BSF, model,
                 first_pred_time, last_pred_time,
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
    output_deaths = []

    for timestep in range(first_pred_time, last_pred_time+1):
        evaluation_deaths = multiindexed_gdf.loc[idx[:, timestep], :]
        evaluation_deaths = evaluation_deaths.drop(columns=timestep_col).reset_index().set_index(location_col)[
            outcome_col]

        #prediction = reg.predict(test_x_BSF[0])
        prediction = reg.predict(test_x_BSF[timestep - first_pred_time])
        output_deaths.append(prediction)
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

    return results_over_time, output_deaths

###

def scikit_model_with_coefficients(multiindexed_gdf, x_BSF, y_BS, test_x_BSF, model,
                                   first_pred_time, last_pred_time,
                                   timestep_col='timestep',
                                   location_col='geoid', outcome_col='deaths',
                                   removed_locations=250, seed=360, bpr_uncertainty_samples=50):

    
    # B = timesteps of data, # S = locations, # F = features
    B, S, F = x_BSF.shape

    # reshape data into 2D for scikit learn models. 1 row = 1 location at 1 time
    x_long = tf.reshape(x_BSF, ((B * S), F))
    y_long = tf.reshape(y_BS, ((B * S), 1))

    reg = model.fit(x_long, tf.squeeze(y_long))
    coefficients = reg.coef_
    print(coefficients)

    # sloppy notation here, it's not the same B
    num_test_times = test_x_BSF.shape[0]

    rng = np.random.default_rng(seed=seed)
    num_sampled = S - removed_locations
    results_over_time = []
    output_deaths = []

    high_prediction_threshold = 50  # arbitrarily set 50

    for timestep in range(first_pred_time, last_pred_time+1):
        evaluation_deaths = multiindexed_gdf.loc[idx[:, timestep], :]
        evaluation_deaths = evaluation_deaths.drop(columns=timestep_col).reset_index().set_index(location_col)[
            outcome_col]

        #prediction = reg.predict(test_x_BSF[0])
        prediction = reg.predict(test_x_BSF[timestep - first_pred_time])
        results_over_samples = []

        for _ in range(bpr_uncertainty_samples):
            sampled_indicies = rng.choice(range(S), size=num_sampled, replace=False)

            results_over_samples.append(
                fast_bpr(evaluation_deaths[sampled_indicies],
                         pd.Series(prediction[sampled_indicies],
                                   index=evaluation_deaths[sampled_indicies].index)
                         )
            )

            #try to change high predictions to 20, for test purposes
            if np.max(prediction[sampled_indicies]) <= high_prediction_threshold:
                output_deaths.append(prediction[sampled_indicies])
            else:
                new_prediction = np.where(prediction[sampled_indicies] <= high_prediction_threshold, 
                                   prediction[sampled_indicies], 40)
                output_deaths.append(new_prediction)

        results_over_time.append(results_over_samples)

    geoids = multiindexed_gdf.index.levels[0]  # Assuming geoid is the first level of the index
    result_data = []

    for geoid in geoids:
        # Extract historical deaths for the current geoid
        evaluation_deaths = multiindexed_gdf.loc[pd.IndexSlice[geoid, :], :]
        evaluation_deaths = evaluation_deaths.droplevel(level='geoid')
        historical_deaths = evaluation_deaths[outcome_col].tolist()

        # Train the model on the concatenated data
        reg = model.fit(x_long, tf.squeeze(y_long))

        # Predictions for the specific geoid using the learned coefficients
        geoid_index = geoids.get_loc(geoid)
        geoid_predictions = [
            reg.predict(test_x_BSF[i - first_pred_time])[geoid_index]
            for i in range(first_pred_time, last_pred_time + 1)
        ]

        # Extract learned coefficients
        coefficients = reg.coef_

        result_data.append({
            'geoid': geoid,
            'historical_deaths': historical_deaths,
            'geoid_predictions': geoid_predictions
        })

    return pd.DataFrame(result_data)


def castnet_model(multiindexed_gdf, dataset_name, cn_result_path, cn_location_path, first_pred_time, last_pred_time, removed_locations=250,
                  timestep_col='timestep', location_col='geoid', outcome_col='deaths', 
                  bpr_uncertainty_samples=50, seed=360):
    """
    Calculate BPR for CASTNet predictions.
    @dataset_name, is either 'cook-county' or 'MA'
    @return: List of BPR results over time and samples
    """
    CN_results = pd.read_csv(cn_result_path)
    CN_results['geoid'] = CN_results['geoid'].astype(str)

    CN_locations = []
    with open(cn_location_path, 'rb') as file:
        for line in file:
            line = line.rstrip().decode("utf-8").split("\t")
            CN_locations.append(line[1])

    # sample and calculate BPR
    rng = np.random.default_rng(seed=seed)
    num_locations = len(CN_locations)
    num_sampled = num_locations - removed_locations
    results_over_time = []
    output_deaths = []

    for timestep in range(first_pred_time, last_pred_time + 1):
        # extract evaluation deaths 
        evaluation_deaths = multiindexed_gdf.loc[idx[:, timestep], :]
        evaluation_deaths = evaluation_deaths.drop(columns=timestep_col).reset_index().set_index(location_col)[outcome_col]

        current_year = 2014 + timestep if dataset_name=='cook-county' else 2000 + timestep 
        predicted_deaths_df = CN_results[(CN_results['year'] == current_year)].set_index('geoid')
        predicted_deaths = predicted_deaths_df['prediction'].values
        output_deaths.append(predicted_deaths)

        results_over_samples = []
        for _ in range(bpr_uncertainty_samples):
            sampled_indices = rng.choice(range(num_locations), size=num_sampled, replace=False)
            evaluation_deaths_series = evaluation_deaths.iloc[sampled_indices]
            predicted_deaths_sampled = predicted_deaths_df.iloc[sampled_indices]['prediction']
            results_over_samples.append(fast_bpr(evaluation_deaths_series, predicted_deaths_sampled))

        results_over_time.append(results_over_samples)

    return results_over_time, output_deaths

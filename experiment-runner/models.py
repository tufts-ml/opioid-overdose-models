import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import IndexSlice as idx
from metrics import fast_bpr
import pickle
import os 

#####
def all_zeroes_model(multiindexed_gdf, first_pred_time, last_pred_time, num_locations, timestep_col='timestep',
                     location_col='geoid', outcome_col='deaths',
                     removed_locations=250, bpr_uncertainty_samples=50, seed=360):

    rng = np.random.default_rng(seed=seed)
    num_sampled = num_locations - removed_locations
    results_over_time = []
    actual_deaths=[]
    predicted_deaths=[]

    for timestep in range(first_pred_time, last_pred_time+1):
        evaluation_deaths = multiindexed_gdf.loc[idx[:, timestep], :]
        evaluation_deaths = evaluation_deaths.drop(columns=timestep_col).reset_index().set_index(location_col)[outcome_col]
        results_over_samples = []

        for _ in range(bpr_uncertainty_samples):
            sampled_indicies = rng.choice(range(num_locations), size=num_sampled, replace=False)
            results_over_samples.append(fast_bpr(evaluation_deaths[sampled_indicies], evaluation_deaths[sampled_indicies]*0))
            actual_deaths.append(evaluation_deaths[sampled_indicies])
            predicted_deaths.append(evaluation_deaths[sampled_indicies]*0)

        results_over_time.append(results_over_samples)

    return results_over_time, actual_deaths, predicted_deaths


def last_time_model(multiindexed_gdf, first_pred_time, last_pred_time, num_locations,
                     pred_lag,
                     timestep_col='timestep',
                     location_col='geoid', outcome_col='deaths',
                     removed_locations=250, bpr_uncertainty_samples=50, seed=360):
    """pred_lag is how many timesteps between the  time used for prediction and evaluation"""

    rng = np.random.default_rng(seed=seed)
    num_sampled = num_locations - removed_locations
    results_over_time = []
    actual_deaths = [] #to store actual values
    output_deaths =[] #to store predictions

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
            output_deaths.append(predicted_deaths[sampled_indicies])
            actual_deaths.append(evaluation_deaths[sampled_indicies])

        results_over_time.append(results_over_samples)

    return results_over_time, actual_deaths, output_deaths

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
    actual_deaths = []

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
            output_deaths.append(predicted_deaths[sampled_indicies])
            actual_deaths.append(evaluation_deaths[sampled_indicies])


        results_over_time.append(results_over_samples)

    return results_over_time, actual_deaths, output_deaths


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
    actual_deaths = []

    high_prediction_threshold = 50  # Adjust this threshold as needed

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

            actual_deaths.append(evaluation_deaths[sampled_indicies]) #append actual death

            #try to change high predictions to 20, for test purposes
            if np.max(prediction[sampled_indicies]) <= high_prediction_threshold:
                output_deaths.append(prediction[sampled_indicies])
            else:
                new_prediction = np.where(prediction[sampled_indicies] <= high_prediction_threshold, 
                                   prediction[sampled_indicies], 40)
                output_deaths.append(new_prediction)

        results_over_time.append(results_over_samples)

    return results_over_time, actual_deaths, output_deaths


###################

#import CASTNet Results 
data_dir = '/Users/jyontika/Desktop/opioid-overdose-models/CASTNet/hughes-CASTNet/'
results_path = os.path.join(data_dir, 'Results/cook-county-predictions.csv') #change to cook-county or MA depending on which you want to run
CN_results = pd.read_csv(results_path)
CN_results['geoid'] = CN_results['geoid'].astype(str)

#import CASTNet locations
locations_path = os.path.join(data_dir, 'Data/Chicago/locations.txt')  #change to Chicago or MA depending on which you want to run

CN_locations = []
with open(locations_path, 'rb') as file:
    for line in file:
        line = line.rstrip().decode("utf-8").split("\t")
        CN_locations.append(line[1])

def castnet_model(multiindexed_gdf, first_pred_time, last_pred_time, num_locations,
                    timestep_col='timestep',
                    location_col='geoid', outcome_col='deaths', removed_locations=250, 
                    bpr_uncertainty_samples=50, seed=360, locations=None):
    """
    Calculate BPR for CASTNet predictions.
    @return: List of BPR results over time and samples
    """
    rng = np.random.default_rng(seed=seed)
    num_sampled = num_locations - removed_locations
    output_deaths=[]
    actual_deaths = []
    results_over_time = []

    for timestep in range(first_pred_time, last_pred_time+1):
        evaluation_deaths = multiindexed_gdf.loc[idx[:, timestep], :]
        evaluation_deaths = evaluation_deaths.drop(columns=timestep_col).reset_index().set_index(location_col)[outcome_col]

        current_year = 2014 + timestep #2000 for MA, 2014 for cook county
        predicted_deaths_df = CN_results[(CN_results['year'] == current_year) & (CN_results['geoid'].isin(CN_locations))]
    

        if CN_locations is not None:
            # Match the order of locations with the order of data
            evaluation_deaths = evaluation_deaths.loc[CN_locations]

        results_over_samples = []

        for _ in range(bpr_uncertainty_samples):
            sampled_indices = rng.choice(range(num_locations), size=num_sampled, replace=False)

            # Convert evaluation_deaths into a pandas Series
            evaluation_deaths_series = pd.Series(evaluation_deaths.iloc[sampled_indices].values, index=sampled_indices)

            # Use predicted_deaths_df for the specific year
            predicted_deaths_sampled = pd.Series(predicted_deaths_df.iloc[sampled_indices]['prediction'].values, 
                                                 index=sampled_indices)
            results_over_samples.append(fast_bpr(evaluation_deaths_series, predicted_deaths_sampled))
            output_deaths.append(predicted_deaths_sampled)
            actual_deaths.append(evaluation_deaths_series)


        results_over_time.append(results_over_samples)

    actual_deaths = np.array(actual_deaths)
    output_deaths = np.array(output_deaths)

    return results_over_time, actual_deaths, output_deaths









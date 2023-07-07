import numpy as np
from pandas import IndexSlice as idx
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
import os
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

from metrics import normcdf, fixed_top_X

def run_adam(model, iterations,
             learning_rate, out_dir,
             test_x, test_y,
             timesteps_per_year, test_years,
             timestep_col, geography_col, outcome_col,
             features_only):
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
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)  # gpflow.optimizers.Adam(learning_rate=0.01)

    starting_timestep = int(test_x[timestep_col].min())

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)

            maes = []
            xtops = []
            for year in range(test_years):
                maes_year = []
                xtop_year = []
                for timestep in range(starting_timestep+year*timesteps_per_year, starting_timestep+year*timesteps_per_year+timesteps_per_year):
                    test_x_time = test_x[test_x[timestep_col] == timestep]
                    test_y_time = test_y[test_y[timestep_col] == timestep]
                    _, _, _, fmean, fvar, gmean, gvar, _, _ = model.build_predict(test_x_time.loc[:, features_only].values)
                    g_cond = tf.math.softplus(fmean * normcdf(gmean)).numpy()
                    pred_df = pd.Series(g_cond.squeeze(), index=test_y_time[geography_col])

                    maes_year.append(mean_absolute_error(test_y_time.deaths, pred_df))
                    xtop_year.append(fixed_top_X(test_y_time.set_index(geography_col)[outcome_col], pred_df, 100))
                maes.append(maes_year)
                xtops.append(xtop_year)
            curr_results = {'iter':step, 'elbo': elbo}
            print(xtops)            
            for y, (mae, xtop) in enumerate(zip(maes, xtops)):
                curr_results[f'mae_{y}'] = copy.deepcopy(np.mean(mae))
                curr_results[f'bpr_100_{y}'] = copy.deepcopy(np.mean([thing[3] for thing in xtop]))
            stat_logs = stat_logs.append(curr_results, ignore_index=True)

            stat_logs.to_csv(stat_path)
            model.savemodel(model_path)

    return logf

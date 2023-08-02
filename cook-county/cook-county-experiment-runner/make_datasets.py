from pandas import IndexSlice as idx
import numpy as np
import tensorflow as tf


def make_data(multiindexed_gdf, first_year, last_year, time_window, feature_cols, num_locations,
              pred_lag=1,
              timesteps_per_year=1,
              year_col='year', timestep_col='timestep', death_col='deaths'):
    """Turn a geodataframe into numpy arrays for model fitting

    :param multiindexed_gdf: Geodataframe, must contain a MultiIndex on [geography, time]
    :param first_year (int): The first year to make predictions for
    :param last_year (int): The final year to make predictions for, can be the same as first_year
    :param time_window (int): How many timesteps of data prior to the prediction year to include
    :param feature_cols (list[str]): The column names to be included as features
    :param num_locations (int): Number of unique locations
    :param pred_lag (int): The number of timesteps between the outcome y and the inputs x. For annual data, simply 1.
        For quarterly data, there could be a 2-4 quarter lag
    :param timesteps_per_year (int): How many timesteps in a year? 1 for year, 4 for quarter, etc.
    :param year_col (str): The name of the column containing the year
    :param timestep_col (str): The neame of the temporal index level
    :param death_col (str): Name of column with deaths
    :return: x_BSF, y_BS: Two tensors. The x array has data for the time_window before the test data, and the
        y array has data for the test years. The x array is shape BSF where B is the number of timesteps in the testing
        years, S is the number of unique locations, and F is the number of features multiplied by the time window.
         y is shape BS
    """
    xs = []
    ys = []
    timesteps = []

    # Iterate over years we want to make predictions for
    for eval_year in range(first_year, last_year + 1):

        timesteps_in_year = multiindexed_gdf[multiindexed_gdf[year_col] == eval_year].index.unique(level=timestep_col)
        timesteps_in_year.sort_values()

        # limit our data to the times and features we care about
        # min(timesteps_in_year) = the first timestep of the first year we want to make predictions for
        # time_window = how many timesteps in our training data

        train_x_df = multiindexed_gdf.loc[
            idx[:, min(timesteps_in_year) - time_window:max(timesteps_in_year) - pred_lag], feature_cols]

        for t, timestep in enumerate(timesteps_in_year):
            #import pdb; pdb.set_trace()
            train_x_vals = train_x_df.values.reshape((num_locations, time_window, len(feature_cols)))

            # Our y data is just the index and deaths
            train_y_df = multiindexed_gdf.loc[idx[:, timestep], death_col]
            train_y_vals = train_y_df.values

            xs.append(train_x_vals)
            ys.append(train_y_vals)
            timesteps.append(np.ones_like(train_y_vals) * t)

    x_BSTD = np.stack(xs, axis=0)
    y_BS = np.stack(ys)

    x_BSTD = tf.convert_to_tensor(x_BSTD, dtype=tf.float32)
    y_BS = tf.convert_to_tensor(y_BS, dtype=tf.float32)

    B, S, T, D = x_BSTD.shape

    assert (B == len(range(first_year, last_year + 1)) * timesteps_per_year)
    assert (S == num_locations)
    assert (T == time_window)
    assert (D == len(feature_cols))

    # Reshape the training data to flatten the dimensions
    x_BSF_flat = tf.reshape(x_BSTD, (B, S, T * D), )

    return x_BSF_flat, y_BS
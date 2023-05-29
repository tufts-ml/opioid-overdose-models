from pandas import IndexSlice as idx
import numpy as np
import tensorflow as tf


def make_data(multiindexed_gdf, first_year, last_year, time_window, feature_cols, train_shape, pred_lag=1):
    xs = []
    ys = []
    quarters = []

    for eval_year in range(first_year, last_year + 1):
        quarters_in_year = multiindexed_gdf[multiindexed_gdf['year'] == eval_year].index.unique(level='timestep')
        quarters_in_year.sort_values()
        train_x_df = multiindexed_gdf.loc[
            idx[:, min(quarters_in_year) - time_window:max(quarters_in_year) - pred_lag], feature_cols]

        for q, quarter in enumerate(quarters_in_year):
            train_x_vals = train_x_df.values.reshape(train_shape)

            train_y_df = multiindexed_gdf.loc[idx[:, quarter], 'deaths']
            train_y_vals = train_y_df.values

            xs.append(train_x_vals)
            ys.append(train_y_vals)
            quarters.append(np.ones_like(train_y_vals) * q)

    x_BSTD = np.stack(xs, axis=0)
    y_BS = np.stack(ys)

    x_BSTD = tf.convert_to_tensor(x_BSTD, dtype=tf.float32)
    y_BS = tf.convert_to_tensor(y_BS, dtype=tf.float32)

    B, S, T, D = x_BSTD.shape

    assert (B == len(range(first_year, last_year + 1)) * pred_lag)
    assert (S == train_shape[0])
    assert (T == time_window)
    assert (D == len(feature_cols))

    # Reshape the training data to flatten the dimensions
    x_BSF_flat = tf.reshape(x_BSTD, (B, S, T * D), )
    # add prediction quarter
    # x_BSF_flat = np.concatenate((x_BSF_flat, np.expand_dims(quarters,axis=-1)),axis=-1)
    times_per_year = len(quarters_in_year)

    if times_per_year > 1:
        # add an indicator
        x_BSF_flat = np.concatenate((x_BSF_flat,
                                     tf.one_hot(np.stack(quarters),
                                                depth=times_per_year, axis=-1)
                                     ), axis=-1)

    return x_BSF_flat, y_BS
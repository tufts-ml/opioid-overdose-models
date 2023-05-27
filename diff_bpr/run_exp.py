import geopandas as gpd

import numpy as np
import tensorflow as tf
import sys
import os

from functools import partial


from top_k import top_k_idx
from make_datasets import make_data
from bpr_model import PerturbedBPRModel

def run_exp(noise=None, perturbation_samples=None, learning_rate=None,
            seed=None, data_path=None, log_dir=None, perturb_code_dir=None):

    sys.path.append(perturb_code_dir)
    from perturbations import perturbed

    epochs = 5000
    time_window = 10
    first_train_eval_year = 2013
    last_train_eval_year = 2017
    batch_dim_size = last_train_eval_year - first_train_eval_year + 1
    validation_year = 2018
    first_test_year = 2019
    last_test_year = 2020

    tf.random.set_seed(seed)


    timestep_col = 'timestep'
    geography_col = 'geoid'
    outcome_col = 'deaths'

    x_idx_cols = [geography_col, 'lat', 'lon', timestep_col,
                  'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc',
                  'svi_pctile', 'year',
                  'neighbor_t', 'self_t-1']
    y_idx_cols = [geography_col, timestep_col, outcome_col]
    """features_only = ['lat', 'lon', timestep_col,
                     'theme_1_pc', 'theme_2_pc', 'theme_3_pc', 'theme_4_pc',
                     'svi_pctile',
                     'neighbor_t', 'self_t-1']"""
    features_only = ['deaths']

    data_gdf = gpd.read_file(data_path)

    multiindexed_gdf = data_gdf.set_index(['geoid', 'year'])
    num_geoids = len(data_gdf['geoid'].unique())

    train_shape = (num_geoids, time_window, len(features_only))

    train_x_BSF_flat, train_y_BS = make_data(multiindexed_gdf, first_train_eval_year, last_train_eval_year,
                                             time_window, features_only, train_shape)

    valid_x_BSF_flat, valid_y_BS = make_data(multiindexed_gdf, validation_year, validation_year,
                                             time_window, features_only, train_shape)

    test_x_BSF_flat, test_y_BS = make_data(multiindexed_gdf, first_test_year, last_test_year,
                                           time_window, features_only, train_shape)

    top_100_idx_func = partial(top_k_idx, k=100)

    perturbed_top_100 = perturbed(top_100_idx_func,
                                  num_samples=perturbation_samples,
                                  sigma=noise,
                                  noise='normal',
                                  batched=True)

    model = PerturbedBPRModel(perturbed_top_100)

    checkpoint_path = os.path.join(log_dir, 'model_{epoch:02d}.hdf5')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        save_freq='epoch',
        initial_value_threshold=-0.35
    )
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model
    def weird_loss(a, b):
        return -a / b

    model.compile(optimizer=optimizer, loss=weird_loss)

    model.fit(train_x_BSF_flat, train_y_BS, epochs=epochs, batch_size=batch_dim_size,
              validation_data=(valid_x_BSF_flat, valid_y_BS),
              callbacks=[checkpoint_callback, tb_callback])

    test_loss = model.evaluate(test_x_BSF_flat, test_y_BS)
    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.scalar('test_loss', test_loss, step=epochs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('-n', '--noise', type=float, help='Noise value', required=True)
    parser.add_argument('-p', '--perturbation_samples', type=int, help='Number of perturbation samples', required=True)
    parser.add_argument('-l', '--learning_rate', type=float, help='Learning rate', required=True)
    parser.add_argument('-s', '--seed', type=int, default=360, help='Seed value (default: 360)')
    parser.add_argument('-d', '--data_path', type=str, help='Data path', required=True)
    parser.add_argument('-ld', '--log_dir', type=str, help='Log directory', required=True)
    parser.add_argument('-pcd', '--perturb_code_dir', type=str, help='Path to perturbations moduel', required=True)

    args = parser.parse_args()
    kwargs = vars(args)  # Convert args to a dictionary
    run_exp(**kwargs)

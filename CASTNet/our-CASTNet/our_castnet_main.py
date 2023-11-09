import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
np.random.seed(401)
import tensorflow as tf
tf.random.set_seed(401)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, auc, roc_curve, roc_auc_score, f1_score
import our_data
import our_castnet_model
import argparse
import sys
import pickle
import helper

class Config:
    hidden_layer_size = 32
    no_epochs = 300
    num_train_years = 13
    num_test_years = 1
    num_valid_years = 1
    window_size = None
    lead_time = None
    time_unit = 1
    group_lasso = False
    dist = None
    gl_reg_coef = None
    rnn_dropout = 0.1
    temporal_feature_size = None
    static_feature_size = None
    no_locations = None
    batch_size = 1000
    learning_rate = 0.005
    embedding_size = 32
    test_time = False
    dataset_name = None
    num_spatial_heads = None
    orthogonal_loss_coef = None

def run(castnet_datadir, cook_county=True):
    conf = Config()

    parser = argparse.ArgumentParser(description='Non-Event or Event')

    parser.add_argument('--ip', default='None', type=str)
    parser.add_argument('--stdin', default='None', type=str)
    parser.add_argument('--control', default='None', type=str)

    parser.add_argument('--dataset_name', default='None', type=str)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--window_size', default=3, type=int)
    parser.add_argument('--lead_time', default=2, type=int)
    parser.add_argument('--gl_reg_coef', default=0.0025, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--group_lasso', default='True', type=str)
    parser.add_argument('--dist', default='root_d', type=str)
    parser.add_argument('--test_time', default='False', type=str)
    parser.add_argument('--num_spatial_heads', default=4, type=int)
    parser.add_argument('--orthogonal_loss_coef', default=0.01, type=float)

    args, _ = parser.parse_known_args()

    conf.dataset_name = "Chicago" if cook_county else "MA"
    conf.hidden_size = args.hidden_size
    conf.window_size = args.window_size
    conf.lead_time = args.lead_time
    conf.gl_reg_coef = args.gl_reg_coef
    conf.dropout = args.dropout
    conf.dist = args.dist
    conf.num_spatial_heads = args.num_spatial_heads
    conf.orthogonal_loss_coef = args.orthogonal_loss_coef
    conf.group_lasso = helper.str2bool(args.group_lasso)
    conf.test_time = helper.str2bool(args.test_time)

    (train_svi_local, train_svi_global, train_static,
      train_sample_indices, train_dist, train_y,
        valid_svi_local, valid_svi_global, valid_static,
          valid_sample_indices, valid_dist, valid_y,
            test_svi_local, test_svi_global, test_static,
              test_sample_indices, test_dist, test_y) = our_data.readData(castnet_datadir, conf.dataset_name, conf.window_size, conf.lead_time,
                                                                              conf.num_train_years, conf.num_test_years, conf.num_valid_years,
                                                                                conf.dist, conf.time_unit)

    if(conf.test_time == True):
        train_svi_global = np.concatenate([train_svi_global, valid_svi_global], axis=0)
        train_svi_local = np.concatenate([train_svi_local, valid_svi_local], axis=0)
        train_static = np.concatenate([train_static, valid_static], axis=0)
        train_sample_indices = np.concatenate([train_sample_indices, valid_sample_indices], axis=0)
        train_dist = np.concatenate([train_dist, valid_dist], axis=0)
        train_y = np.concatenate([train_y, valid_y], axis=0)

    train_svi_global = np.swapaxes(train_svi_global, 1, 2)
    valid_svi_global = np.swapaxes(valid_svi_global, 1, 2)
    test_svi_global = np.swapaxes(test_svi_global, 1, 2)

    temporal_feature_size = train_svi_global.shape[-1]
    no_locations = train_svi_global.shape[2]
    static_feature_size = train_static.shape[1]

    conf.temporal_feature_size = temporal_feature_size
    conf.no_locations = no_locations
    conf.static_feature_size = static_feature_size

    sess = tf.compat.v1.Session()

    castnet = our_castnet_model.CASTNet(sess=sess, conf=conf)
    print(train_svi_local.shape[0])
    castnet.train(train_svi_local, train_svi_global, train_static, train_sample_indices, train_dist, train_y, valid_svi_local, valid_svi_global, valid_static, valid_sample_indices, valid_dist, valid_y, test_svi_local, test_svi_global, test_static, test_sample_indices, test_dist, test_y)

    data_dir = os.environ.get('DATA_DIR', "./")
    model_save_path = os.path.join(data_dir, 'Results')
    
    if cook_county:
        model_save_path = os.path.join(model_save_path, 'Chicago')
    else:
        model_save_path = os.path.join(model_save_path, 'MA')

    castnet.save(model_save_path)
    castnet.save_results()

### to use ? 
if __name__ == '__main__':
    run(cook_county=True)  # set to False if you want to save MA results
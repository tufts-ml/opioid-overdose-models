import numpy as np
import pandas as pd
import tensorflow as tf
def normcdf(x):
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1. - 2.e-3) + 1.e-3


def fixed_top_X(true_qtr_val, pred_qtr_val, X=10):
    top_X_predicted = pred_qtr_val.sort_values(ascending=False).iloc[:X]
    top_X_true = true_qtr_val.sort_values(ascending=False).iloc[:X]

    undisputed_top_predicted = top_X_predicted[top_X_predicted > top_X_predicted.min()]
    num_tied_spots = X - len(undisputed_top_predicted)
    undisputed_top_true = top_X_true[top_X_true > top_X_true.min()]
    num_true_ties = X - len(undisputed_top_true)

    tied_top_predicted = pred_qtr_val[pred_qtr_val == top_X_predicted.min()]
    tied_top_true = true_qtr_val[true_qtr_val == top_X_true.min()]

    error_in_top_true_ties = np.abs(tied_top_true - pred_qtr_val[tied_top_true.index]).sort_values(ascending=True)
    error_in_top_pred_ties = np.abs(true_qtr_val[tied_top_predicted.index] - tied_top_predicted).sort_values(
        ascending=True)
    top_true_tied_geoids = error_in_top_true_ties[:num_true_ties].index
    top_pred_tied_geoids = error_in_top_pred_ties[:num_tied_spots].index

    best_possible_top_true_geoids = pd.Index.union(undisputed_top_true.index, top_true_tied_geoids)
    best_possible_top_pred_geoids = pd.Index.union(undisputed_top_predicted.index, top_pred_tied_geoids)

    # True values of GEOIDS with highest actual deaths. If ties, finds tied locations that match preds best
    best_possible_true = true_qtr_val[best_possible_top_true_geoids]
    best_possible_pred = true_qtr_val[best_possible_top_pred_geoids]

    assert (len(best_possible_true) == X)
    assert (len(best_possible_pred) == X)

    best_possible_absolute = np.abs(best_possible_true.sum() - best_possible_pred.sum())
    best_possible_ratio = np.abs(best_possible_pred).sum() / np.abs(best_possible_true).sum()

    bootstrapped_tied_indices = np.random.choice(tied_top_predicted.index, (1000, num_tied_spots))
    bootstrapped_all_indices = [pd.Index.union(undisputed_top_predicted.index,
                                               bootstrap_index) for bootstrap_index in bootstrapped_tied_indices]

    bootstrapped_absolute = np.mean([np.abs(top_X_true.sum() - true_qtr_val[indices].sum())
                                     for indices in bootstrapped_all_indices])
    bootstrapped_ratio = np.mean([np.abs(true_qtr_val[indices]).sum() / np.abs(top_X_true).sum()
                                  for indices in bootstrapped_all_indices])

    return best_possible_absolute, best_possible_ratio, bootstrapped_absolute, bootstrapped_ratio

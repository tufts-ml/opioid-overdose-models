import numpy as np

def fast_bpr(true_val, pred_val, K=100, bootstrap_samples=1000):
    """

    :param true_val: Pandas dataframe indexed on location
    :param pred_val: Pandas dataframe indexed on location
    :param K: Number of locations to consider
    :param bootstrap_samples: Number of samples to take when evaluating ties
    :return:
    """
    top_K_predicted = pred_val.sort_values(ascending=False).iloc[:K]
    top_K_true = true_val.sort_values(ascending=False).iloc[:K]

    # Now we check for ties
    undisputed_top_predicted = top_K_predicted[top_K_predicted > top_K_predicted.min()]
    num_tied_spots = K - len(undisputed_top_predicted)
    undisputed_top_true = top_K_true[top_K_true > top_K_true.min()]
    num_true_ties = K - len(undisputed_top_true)

    tied_top_predicted = pred_val[pred_val == top_K_predicted.min()]
    tied_top_true = true_val[true_val == top_K_true.min()]

    # now randomly choose locations from the tied spots
    bootstrapped_tied_indices = np.random.choice(tied_top_predicted.index, (bootstrap_samples, num_tied_spots))
    undisputed_pred_idx = undisputed_top_predicted.index.values
    bootstrapped_all_indices = [np.concatenate((undisputed_pred_idx, bootstrap_index))
                                for bootstrap_index in bootstrapped_tied_indices]


    denominator =  top_K_true.sum()
    numerators = [true_val[indicies].sum() for indicies in bootstrapped_all_indices]

    bootstrapped_ratio = np.mean([numerator / denominator
                                  for numerator in numerators])

    return bootstrapped_ratio
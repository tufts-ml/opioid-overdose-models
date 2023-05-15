from functools import partial
import tensorflow as tf
def bpr_variable_k_no_ties(y_true, y_pred, k=None):
    """Calculate BPR-k often not used due to loss functions expected to only take 2 arguments

    Args:
        y_true: True outcome
        y_pred: predicted outcome
        k (int): Threshold for BPR. Required, but defaults to None because partial re

    Returns:
        The BPR-k score

    Note: This method DOES NOT handle ties, as it is meant to be used in a perturbed fashion
    """

    _, top_k_pred_idx = tf.math.top_k(y_pred, k=k)
    top_k_true_val, top_k_true_idx = tf.math.top_k(y_true, k=k)

    # Denominator is actual top-k
    # Impossible to have ties here, a tie wouldn't change the value
    denominator = tf.reduce_sum(top_k_true_val, axis=-1)

    # Numerator is sum of true values at the locations indicated by predictions
    # Note: there could be ties here. We choose to ignore and deal with noise
    true_val_at_pred_top_k = tf.gather(y_true, top_k_pred_idx, batch_dims=-1)
    numerator = tf.reduce_sum(true_val_at_pred_top_k, axis=-1)

    bpr_k_value = numerator / denominator

    return bpr_k_value






import tensorflow as tf

def top_k_idx(input_BD, **kwargs):
    _, idx_BD = tf.math.top_k(input_BD, **kwargs)
    input_depth = input_BD.shape[-1]
    one_hot_idx_BKD = tf.one_hot(idx_BD, input_depth)
    # Sum over k dimension so we dont have to worry about sorting
    k_hot_idx_BD = tf.reduce_sum(one_hot_idx_BKD, axis=-2)

    return k_hot_idx_BD

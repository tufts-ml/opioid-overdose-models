"""Tests for differentiable bpr."""

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

import bpr


class BPRTest(parameterized.TestCase, tf.test.TestCase):
    """Testing the bpr module."""

    def setUp(self):
        super(BPRTest, self).setUp()
        tf.random.set_seed(0)
        self.y_true = tf.constant([1, 2, 3, 4, 5])
        self.y_pred_all_wrong = tf.constant([50, 40, 30, 20, 10])
        self.y_pred_1_right = tf.constant([40, 30, 20, 10, 50])
        self.y_pred_2_right = tf.constant([.3, .2, .1, .5, .4])
        self.y_pred_3_right = tf.constant([-40, -50, -10, -20, -30])
        self.y_pred_all_right = tf.constant([-5000, -50, 0, 50, 100])

        self.y_true_batch = tf.constant([[1,2,3],
                                         [3,2,1]])
        self.y_pred_1_right_batch = tf.constant([[20, 10, 30],
                                                 [-1, -10, -5]])

    @parameterized.parameters([1, 2, 3, 4, 5])
    def test_bpr_right_order(self, k):
        bpr_wrong = bpr.bpr_variable_k_no_ties(self.y_true, self.y_pred_all_wrong, k)
        bpr_one_right = bpr.bpr_variable_k_no_ties(self.y_true, self.y_pred_1_right, k)
        bpr_all_right = bpr.bpr_variable_k_no_ties(self.y_true, self.y_pred_all_right, k)

        # When every element is wrong, the predictions will say that the bottom k are the top
        neg_bottom_k_val, _ =  tf.math.top_k(-self.y_true, k=k)
        top_k_val, _ = tf.math.top_k(self.y_true, k=k)
        wrong_numerator = tf.reduce_sum(-neg_bottom_k_val)

        one_right_numerator = tf.reduce_max(self.y_true) + tf.reduce_sum(tf.sort(self.y_true)[:k - 1])
        all_right_numerator = tf.reduce_sum(tf.sort(self.y_true, direction='DESCENDING')[:k])

        denominator = tf.reduce_sum(top_k_val)

        calc_wrong_bpr = wrong_numerator/denominator
        calc_one_right = one_right_numerator / denominator
        calc_all_right = all_right_numerator / denominator

        self.assertEqual(calc_wrong_bpr, bpr_wrong)
        self.assertEqual(calc_one_right, bpr_one_right)
        self.assertEqual(calc_all_right, bpr_all_right)

    @parameterized.parameters([1, 2, 3, 4, 5])
    def test_bpr_wrong_order(self, k):

        bpr_two_right = bpr.bpr_variable_k_no_ties(self.y_true, self.y_pred_2_right, k)
        bpr_three_right = bpr.bpr_variable_k_no_ties(self.y_true, self.y_pred_3_right, k)

        denominator = tf.reduce_sum(tf.sort(self.y_true, direction='DESCENDING')[:k])

        num_correct = min(k, 2)
        num_incorrect = max(0, k-2)
        two_right_numerator = tf.reduce_sum(tf.sort(self.y_true, direction='DESCENDING')[:num_correct]) + \
                              tf.reduce_sum(tf.sort(self.y_true)[:num_incorrect])
        calculated_two_right = two_right_numerator / denominator

        num_correct = min(k, 3)
        num_incorrect = max(0, k - 3)
        three_right_numerator = tf.reduce_sum(tf.sort(self.y_true, direction='DESCENDING')[:num_correct]) + \
                                tf.reduce_sum(tf.sort(self.y_true)[:num_incorrect])
        calculated_three_right = three_right_numerator / denominator

        # For k >= the number of correct predictions, order wont matter
        if k < 2:
            self.assertNotEqual(bpr_two_right, calculated_two_right)
        else:
            self.assertEqual(bpr_two_right, calculated_two_right)

        if k < 3:
            self.assertNotEqual(bpr_three_right, calculated_three_right)
        else:
            self.assertEqual(bpr_three_right, calculated_three_right)

    @parameterized.parameters([1, 2, 3])
    def test_bpr_batch(self, k):
        bpr_one_right = bpr.bpr_variable_k_no_ties(self.y_true_batch,
                                                   self.y_pred_1_right_batch, k)

        # When every element is wrong, the predictions will say that the bottom k are the top
        one_right_numerator = tf.reduce_max(self.y_true_batch, axis=-1) + \
                              tf.reduce_sum(tf.sort(self.y_true_batch, axis=-1)[:, :k-1], axis=-1)


        top_k_val, _ = tf.math.top_k(self.y_true_batch, k=k)
        denominator = tf.reduce_sum(top_k_val, axis=-1)

        calc_one_right = one_right_numerator / denominator

        self.assertAllEqual(calc_one_right, bpr_one_right)


if __name__ == '__main__':
    tf.enable_v2_behavior()
    tf.test.main()

import numpy as np
import sklearn.linear_model
import json
from make_xy_data_splits import make_outcome_history_feat_names

def make_hyper_grid(Wmax=4, **kwargs):
    return {}

def construct(*args, **kwargs):
    return ZeroPred(*args, **kwargs)

class ZeroPred(object):

    def __init__(self, ):
        '''
        '''
        self.__name__ = 'ZeroPred'

    @classmethod
    def get_hyper_grid(cls):
        return make_hyper_grid()

    def get_params(self):
        return {}

    def set_params(self, **kwargs):
        pass

    def save_params(self, fname):
        with open(fname, 'w') as f:
            json.dump(self.get_params(), f)

    def fit(self, tr_x_df, tr_y_df):
        '''
        '''
        return self


    def predict(self, x_df):
        '''
        '''
        return np.zeros(len(x_df))

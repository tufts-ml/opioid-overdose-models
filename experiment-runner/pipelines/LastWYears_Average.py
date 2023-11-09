import numpy as np
import json

def make_hyper_grid(Wmax=4, **kwargs):
    return {
            'W': np.arange(1, Wmax+1), # Want Wmax to be inclusive
        }

def construct(*args, **kwargs):
    return AvgLastWYears(*args, **kwargs)

class AvgLastWYears(object):

    def __init__(self, W):
        '''
        '''
        self.W = W
        self.__name__ = 'AvgLastWYears'

    @classmethod
    def get_hyper_grid(cls):
        return make_hyper_grid()

    def get_params(self):
        return {'W': int(self.W)}

    def set_params(self, **kwargs):
        if 'W' in kwargs:
            self.W = kwargs['W']

    def save_params(self, fname):
        with open(fname, 'w') as f:
            json.dump(self.get_params(), f)

    def fit(self, tr_x_df, tr_y_df):
        '''
        '''
        self._train_x_df = tr_x_df
        self._train_y_df = tr_y_df

    def predict(self, x_df):
        '''
        '''        
        pred_cols = [f'prev_deaths_{w:02d}back' for w in range(1,self.W+1)]
        # take average over x_df from pred_cols
        yhat_Q = x_df[pred_cols].mean(axis=1).values

        return yhat_Q

import numpy as np
import sklearn.linear_model
import json
from make_xy_data_splits import make_outcome_history_feat_names

def make_hyper_grid(Wmax=4, **kwargs):
    fnames = make_outcome_history_feat_names(Wmax)

    if 'added_cols' in kwargs:
        feat_cols = [fnames[x:] + kwargs['added_cols'] for x in range(Wmax)]

    return {
            'alpha': np.logspace(-6, 8, 15 + 14),
            'feat_cols': feat_cols,
            'max_iter': [1000],
        }

def construct(*args, **kwargs):
    return LinearRegressor(*args, **kwargs)

class LinearRegressor(object):

    def __init__(self, alpha=0.0, feat_cols=None, max_iter=100):
        '''
        '''
        self.alpha = float(alpha)
        self.feat_cols = feat_cols
        self.max_iter = max_iter
        self.__name__ = 'LinearRegressor'

    @classmethod
    def get_hyper_grid(cls):
        return make_hyper_grid()

    def get_params(self):
        return {'feat_cols':self.feat_cols, 'alpha':self.alpha, 'max_iter':self.max_iter}

    def set_params(self, **kwargs):
        if 'feat_cols' in kwargs:
            self.feat_cols = kwargs['feat_cols']
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        if 'max_iter' in kwargs:
            self.max_iter = kwargs['max_iter']

    def save_params(self, fname):
        with open(fname, 'w') as f:
            json.dump(self.get_params(), f)

    def _txfm_x(self, x_df):
        if self.feat_cols is None:
            return x_df.values
        else:
            return x_df[self.feat_cols].values

    def fit(self, tr_x_df, tr_y_df):
        '''
        '''
        self._skregr = sklearn.linear_model.Ridge(alpha=self.alpha, max_iter=self.max_iter)
        self._skregr.fit(self._txfm_x(tr_x_df), np.squeeze(tr_y_df.values))
        return self


    def predict(self, x_df):
        '''
        '''
        return self._skregr.predict(self._txfm_x(x_df))

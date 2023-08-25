import numpy as np
import sklearn.linear_model
from make_xy_data_splits import make_outcome_history_feat_names

def make_hyper_grid(Wmax=4, **kwargs):
    fnames = make_outcome_history_feat_names(Wmax)
    return {
            'alpha': np.logspace(-6, 8, 15 + 14),
            'feat_cols': [fnames[x:] for x in range(Wmax)],
        }

def construct(*args, **kwargs):
    return LinearRegressor(*args, **kwargs)

class LinearRegressor(object):

    def __init__(self, alpha=0.0, feat_cols=None):
        '''
        '''
        self.alpha = float(alpha)
        self.feat_cols = feat_cols
        self.__name__ = 'LinearRegressor'

    @classmethod
    def get_hyper_grid(cls):
        return make_hyper_grid()

    def get_params(self):
        return {'feat_cols':self.feat_cols, 'alpha':self.alpha}

    def set_params(self, **kwargs):
        if 'feat_cols' in kwargs:
            self.feat_cols = kwargs['feat_cols']
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']

    def _txfm_x(self, x_df):
        if self.feat_cols is None:
            return x_df.values
        else:
            return x_df[self.feat_cols].values

    def fit(self, tr_x_df, tr_y_df):
        '''
        '''
        self._skregr = sklearn.linear_model.Ridge(alpha=self.alpha)
        self._skregr.fit(self._txfm_x(tr_x_df), np.squeeze(tr_y_df.values))
        return self


    def predict(self, x_df):
        '''
        '''
        return self._skregr.predict(self._txfm_x(x_df))

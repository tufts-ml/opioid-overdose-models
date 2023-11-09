import numpy as np
import json
import sklearn.ensemble
from make_xy_data_splits import make_outcome_history_feat_names
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor


def make_hyper_grid(Wmax=4, **kwargs):

    if 'added_cols' in kwargs:
        feat_cols = kwargs['added_cols']

    return {
            'init_length_scale': [0.5],
            'feat_cols': [feat_cols],
        }

def construct(*args, **kwargs):
    return GPRegr(*args, **kwargs)

class GPRegr(object):

    def __init__(self, init_length_scale=0.5, feat_cols=None):
        '''
        '''
        self.init_length_scale = init_length_scale
        self.feat_cols = feat_cols
        self.__name__ = 'GPRegr'

    @classmethod
    def get_hyper_grid(cls):
        return make_hyper_grid()
    
    def _txfm_x(self, x_df):
        if self.feat_cols is None:
            return x_df.values
        else:
            return x_df[self.feat_cols].values


    def get_params(self):
        return {'init_length_scale':self.init_length_scale}

    def set_params(self, **kwargs):
        cur_kws = self.get_params()
        for k, v in kwargs.items():
            if k in cur_kws: # only set attribs that are actually used
                setattr(self, k, v)

    def save_params(self, fname):
        with open(fname, 'w') as f:
            json.dump(self.get_params(), f)

    def fit(self, tr_x_df, tr_y_df):
        '''
        '''
        self.kernel = RBF(length_scale = self._txfm_x(tr_x_df).shape[-1]*[self.init_length_scale]) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-5, 1e1))
        self._gaussian_process = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=9, normalize_y=True)
        self._gaussian_process.fit(self._txfm_x(tr_x_df), np.squeeze(tr_y_df.values))
        return self


    def predict(self, x_df):
        '''
        '''
        return self._gaussian_process.predict(self._txfm_x(x_df))
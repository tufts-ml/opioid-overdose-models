import numpy as np
import sklearn.ensemble
from make_xy_data_splits import make_outcome_history_feat_names
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor


def make_hyper_grid(Wmax=4, **kwargs):
    return {
            'init_length_scale': np.logspace(0, 1, 3, base=10)
        }

def construct(*args, **kwargs):
    return GPRegr(*args, **kwargs)

class GPRegr(object):

    def __init__(self, init_length_scale=0.5):
        '''
        '''
        self.init_length_scale = init_length_scale
        self.__name__ = 'GPRegr'

    @classmethod
    def get_hyper_grid(cls):
        return make_hyper_grid()

    def get_params(self):
        return {'init_length_scale':self.init_length_scale}

    def set_params(self, **kwargs):
        cur_kws = self.get_params()
        for k, v in kwargs.items():
            if k in cur_kws: # only set attribs that are actually used
                setattr(self, k, v)

    def fit(self, tr_x_df, tr_y_df):
        '''
        '''
        self.kernel = RBF(length_scale = tr_x_df.shape[-1]*[self.init_length_scale])
        self._gaussian_process = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=9)
        self._gaussian_process.fit(tr_x_df.values, np.squeeze(tr_y_df.values))
        return self


    def predict(self, x_df):
        '''
        '''
        return self._gaussian_process.predict(x_df.values)
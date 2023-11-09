import numpy as np
import sklearn.linear_model

from pipelines.LinearRegr import LinearRegressor, make_hyper_grid

def construct(*args, **kwargs):
    return PoissonRegressor(*args, **kwargs)

class PoissonRegressor(LinearRegressor):

    def __init__(self, alpha=0.0, feat_cols=None, max_iter=100):
        '''
        '''
        self.alpha = float(alpha)
        self.feat_cols = feat_cols
        self.max_iter = max_iter
        self.__name__ = 'PoissonRegressor'

    @classmethod
    def get_hyper_grid(cls):
        return make_hyper_grid()

    def fit(self, tr_x_df, tr_y_df):
        '''
        '''
        self._skregr = sklearn.linear_model.PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)
        self._skregr.fit(self._txfm_x(tr_x_df), np.squeeze(tr_y_df.values))
        return self

    # predict is inherited from LinearRegr
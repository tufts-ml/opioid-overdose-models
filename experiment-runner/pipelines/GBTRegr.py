import numpy as np
import sklearn.ensemble
from make_xy_data_splits import make_outcome_history_feat_names

def make_hyper_grid(Wmax=4, **kwargs):
    return {
            'loss': ['squared_error', 'poisson'],
            'max_iter': [32, 128],
            'min_samples_leaf': np.logspace(0, 8, 9, base=2).astype(np.int32),
            'max_leaf_nodes': np.logspace(4, 8, 5, base=2).astype(np.int32),
        }

def construct(*args, **kwargs):
    return GBTRegr(*args, **kwargs)

class GBTRegr(object):

    def __init__(self, loss='poisson', max_iter=128, min_samples_leaf=256, max_leaf_nodes=32):
        '''
        '''
        self.loss = loss
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.__name__ = 'GBTRegr'

    @classmethod
    def get_hyper_grid(cls):
        return make_hyper_grid()

    def get_params(self):
        return {'loss':self.loss, 'max_iter':self.max_iter,
                'min_samples_leaf':self.min_samples_leaf,
                'max_leaf_nodes':self.max_leaf_nodes}

    def set_params(self, **kwargs):
        cur_kws = self.get_params()
        for k, v in kwargs.items():
            if k in cur_kws: # only set attribs that are actually used
                setattr(self, k, v)

    def fit(self, tr_x_df, tr_y_df):
        '''
        '''
        self._skregr = sklearn.ensemble.HistGradientBoostingRegressor(
            loss=self.loss,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes)
        self._skregr.fit(tr_x_df.values, np.squeeze(tr_y_df.values))
        return self


    def predict(self, x_df):
        '''
        '''
        return self._skregr.predict(x_df.values)
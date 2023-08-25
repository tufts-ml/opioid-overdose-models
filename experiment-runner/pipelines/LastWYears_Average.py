import numpy as np

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
        return {'W':self.W}

    def set_params(self, **kwargs):
        if 'W' in kwargs:
            self.W = kwargs['W']

    def fit(self, tr_x_df, tr_y_df):
        '''
        '''
        self._train_x_df = tr_x_df
        self._train_y_df = tr_y_df

    def predict(self, x_df):
        '''
        '''
        # search for W rows of _train that match x_df loc and closest in time
        toarr = np.asarray
        geo_Q = toarr(x_df.index.get_level_values('geoid'))
        geo_N = toarr(self._train_x_df.index.get_level_values('geoid'))
        t_Q = toarr(x_df.index.get_level_values('timestep'))
        t_N = toarr(self._train_x_df.index.get_level_values('timestep'))

        dist_NQ = (
            100 * np.abs(geo_N[:,np.newaxis] - geo_Q[np.newaxis,:])
            + np.abs(t_N[:,np.newaxis] - t_Q[np.newaxis,:])
            )
        sorted_ids_NQ = np.argsort(dist_NQ, axis=0)

        W = self.W
        yhat_Q = np.squeeze(
            self._train_y_df.values[sorted_ids_NQ[:W, :]].mean(axis=0))
        return yhat_Q

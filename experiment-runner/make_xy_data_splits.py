import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
import geopandas as gpd
import os

from collections import namedtuple

DataSet = namedtuple('DataSplit', ['x', 'y', 'info'])

SVI_COLS = [
    'svi_theme1_pctile', 'svi_theme2_pctile', 'svi_theme3_pctile',
    'svi_theme4_pctile', 'svi_total_pctile']

LATLON_COLS = [
    'INTPTLAT', 'INTPTLON']

def make_outcome_history_feat_names(W, outcome_col='deaths'):
    return ['prev_%s_%02dback' % (outcome_col, W - ww) for ww in range(W)]

def load_xy_splits(
        data_dir = os.environ.get('DATA_DIR'),
        timescale='year',
        csv_pattern_str='cook_county_gdf_cleanwithsvi_{timescale}.csv',
        timestep_col='timestep',
        geography_col='geoid',
        outcome_col='deaths',
        year_col='year',
        context_size_in_tsteps=3,
        train_years=[2017, 2018, 2019],
        valid_years=[2020],
        test_years=[2021, 2022],
        space_cols=LATLON_COLS,
        svi_cols=SVI_COLS,
        add_space=True,
        add_time=True,
        add_svi=True,
        **kwargs
        ):
    all_df = gpd.read_file(os.path.join(
        data_dir, csv_pattern_str.format(timescale=timescale)))

    x_cols_only = []
    if add_space:
        x_cols_only += space_cols
    if add_time:
        x_cols_only += [timestep_col]
    if add_svi:
        x_cols_only += svi_cols
    y_cols_only = ['deaths']
    info_cols_only = [year_col]

    # Create the multiindex, reinserting timestep as a col not just index
    all_df = all_df.astype({geography_col: int, timestep_col: int})
    mi_df = all_df.set_index([geography_col, timestep_col])
    mi_df[timestep_col] = mi_df.index.get_level_values(timestep_col)

    x_df = mi_df[x_cols_only].copy()
    y_df = mi_df[y_cols_only].copy()
    info_df = mi_df[info_cols_only].copy()

    tr_years = np.sort(np.unique(np.asarray(train_years)))
    va_years = np.sort(np.unique(np.asarray(valid_years)))
    te_years = np.sort(np.unique(np.asarray(test_years)))
    assert np.max(tr_years) < np.min(va_years)
    assert np.max(va_years) < np.min(te_years)
    W = int(context_size_in_tsteps)

    kws = dict(timestep_col=timestep_col,
        year_col=year_col, outcome_col=outcome_col, 
        **kwargs)
    tr_tup = make_x_y_i_data_with_filled_context(
        x_df, y_df, info_df, tr_years[0], tr_years[-1], W, **kws)
    va_tup = make_x_y_i_data_with_filled_context(
        x_df, y_df, info_df, va_years[0], va_years[-1], W, **kws)
    te_tup = make_x_y_i_data_with_filled_context(
        x_df, y_df, info_df, te_years[0], te_years[-1], W, **kws)

    return tr_tup, va_tup, te_tup


def make_x_y_i_data_with_filled_context(
        x_df, y_df, info_df,
        first_year, last_year,
        context_size_in_tsteps,
        lag_in_tsteps=1,
        how_to_handle_tstep_without_enough_context='raise_error',
        year_col='year', timestep_col='timestep', outcome_col='deaths'):
    """ Create x,y,i dataframes suitable for supervised learning

    Fill in features in x corresponding to previously seen y vals as context

    Args
    ----
    x_df
    y_df
    info_df
    first_year : int
        The first year to make predictions for
    last_year : int
        The final year (inclusive) to make predictions for
        Can be the same as first_year
    window_size_in_tsteps : int
        How many timesteps of data prior to the prediction tstep to include
    lag_in_tsteps : int
        The number of timesteps between the outcome y and the inputs x. 
        For example, if you want 3-step-ahead predictions
    year_col (str): The name of the column containing the year
    timestep_col (str): The neame of the temporal index level
    outcome_col (str): Name of column with outcome variable (deaths) we are trying to predict

    Returns
    -------
    x_df : dataframe with shape (N, F)
        Each entry is a valid covariate for predicting y in corresp row
    y_df : dataframe with shape (N, 1)
    i_df : dataframe with shape (N, K)
        Holds "info" columns corresponding to rows in x,y
    """
    first_year = int(first_year)
    last_year = int(last_year)
    assert last_year >= first_year
    
    L = int(lag_in_tsteps)
    W = int(context_size_in_tsteps)
    new_col_names = make_outcome_history_feat_names(W)

    xs = []
    ys = []
    infos = []

    # Iterate over years we want to make predictions for
    for eval_year in range(first_year, last_year + 1):

        t_index = info_df[info_df[year_col] == eval_year].index
        timesteps_in_year = t_index.unique(level=timestep_col).values
        timesteps_in_year = np.sort(np.unique(timesteps_in_year))
        
        for tt, tstep in enumerate(timesteps_in_year):
            # Make per-tstep dataframes
            x_tt_df = x_df.loc[idx[:, tstep], :].copy()
            y_tt_df = y_df.loc[idx[:, tstep], :].copy()
            info_tt_df = info_df.loc[idx[:, tstep], :].copy()

            # Determine if we can get a full window of 'actual' data, or if we need to zero-pad
            if tstep - (W + L - 1) <= 0:
                if how_to_handle_tstep_without_enough_context == 'raise_error':
                    raise ValueError("Not enough context available for tstep %d. Need at least %d previous tsteps" % (tstep, W+L-1))
                assert how_to_handle_tstep_without_enough_context == 'pad_with_zero'
                WW = tstep - L
            else:
                WW = W
            # Grab current tstep's history from outcomes at previous tsteps
            xhist_N = y_df.loc[idx[:, tstep-(WW+L-1):(tstep-L)], outcome_col].values.copy()
            N = xhist_N.shape[0]
            M = N // WW
            xhist_MW = xhist_N.reshape((M, WW))
            if WW < W:
                xhist_MW = np.hstack([ np.zeros((M, W-WW)), xhist_MW])
            assert xhist_MW.shape[1] == W
            for ww in range(W):
                x_tt_df[new_col_names[ww]] = xhist_MW[:, ww]
                
            xs.append(x_tt_df)
            ys.append(y_tt_df)
            infos.append(info_tt_df)

    return  DataSet(pd.concat(xs), pd.concat(ys), pd.concat(infos))
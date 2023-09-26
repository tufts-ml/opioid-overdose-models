import argparse
import numpy as np
import pandas as pd
import os
import itertools
import make_xy_data_splits

from sklearn.metrics import mean_absolute_error, mean_squared_error
from pipelines import LastYear, LastWYears_Average, PoissonRegr, LinearRegr, GBTRegr, GPRegr


def calc_score(model, x_df, y_df, metric):
    return calc_score_dict(model, x_df, y_df, '')[metric]

def calc_score_dict(model, x_df, y_df, split_name, metrics=['mae', 'rmse']):
    ytrue = y_df.values
    yhat = model.predict(x_df)

    for metric in metrics:
        mae = mean_absolute_error(ytrue, yhat)
        rmse = np.sqrt(mean_squared_error(ytrue, yhat))

        return dict(
            mae=mae,
            rmse=rmse,
            neg_mae=-1.0*mae,
            neg_rmse=-1.0*rmse,
            max_yhat=np.max(yhat),
            method=model.__name__,
            hypers=str(model.get_params()),
            split=split_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--location', choices=['MA', 'cook'], help='Which location to run')
    args = parser.parse_args()

    verbose = False

    if args.location == 'MA':
        timescale = 'annual'
        csv_pattern_str = 'clean_{timescale}_tract'
        context_size_in_tsteps = 5
        train_years=[2014, 2015, 2016, 2017, 2018]
        valid_years=[2019]
        test_years=[2020, 2021]
        svi_cols = ['theme_1_pc', 'theme_2_pc', 'theme_3_pc',
                     'theme_4_pc', 'svi_pctile']
        space_cols =  ['lat', 'lon']

    elif args.location =='cook':
        timescale ='year'
        csv_pattern_str = 'cook_county_gdf_cleanwithsvi_{timescale}.csv'
        context_size_in_tsteps = 3
        train_years=[2017, 2018, 2019]
        valid_years=[2020]
        test_years=[2021, 2022]
        svi_cols = [
            'svi_theme1_pctile', 'svi_theme2_pctile', 'svi_theme3_pctile',
            'svi_theme4_pctile', 'svi_total_pctile']

        space_cols = [
            'INTPTLAT', 'INTPTLON']

    context_size_in_tsteps = 3

    tr, va, te = make_xy_data_splits.load_xy_splits(
        timescale=timescale,
        csv_pattern_str=csv_pattern_str,
        train_years=train_years,
        valid_years=valid_years,
        test_years=test_years,
        context_size_in_tsteps=context_size_in_tsteps,
        how_to_handle_tstep_without_enough_context='pad_with_zero',
        svi_cols=svi_cols,
        space_cols=space_cols,
        add_space=True,
        add_time=True,
        add_svi=True)
    
    for model_module in [GPRegr]:

        hyper_grid = model_module.make_hyper_grid(
            Wmax=context_size_in_tsteps)
        keys = hyper_grid.keys()
        vals = hyper_grid.values()
        row_dict_list = list()

        best_score = -np.inf
        best_hypers = None
        best_model = None
        for hyper_vals in itertools.product(*vals):
            hypers = dict(zip(keys, hyper_vals))
            model = model_module.construct(**hypers)
            model.fit(tr.x, tr.y)

            score = calc_score(model, va.x, va.y, 'neg_mae')
            row_dict = dict(**hypers)
            row_dict['score'] = score
            if score > best_score:
                row_dict['winner'] = 1
                best_model = model
                best_hypers = hypers
                best_score = score

            row_dict_list.append(row_dict)

        hyper_df = pd.DataFrame(row_dict_list)
        if verbose:
            print(hyper_df)
        for k, v in best_model.get_params().items():
            assert k in best_hypers
            assert best_hypers[k] == v
        # Record perf on val and test
        va_score_dict = calc_score_dict(best_model, va.x, va.y, 'valid')
        te_score_dict = calc_score_dict(best_model, te.x, te.y, 'test')
        df = pd.DataFrame([va_score_dict, te_score_dict])\
            [['method', 'mae', 'rmse', 'hypers', 'split']].copy()
        print(df)
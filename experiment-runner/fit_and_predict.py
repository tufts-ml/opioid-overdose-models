import argparse
import numpy as np
import pandas as pd
import os
import itertools
import make_xy_data_splits

from sklearn.metrics import mean_absolute_error, mean_squared_error
import pipelines
from pipelines import LastYear, LastWYears_Average, PoissonRegr, LinearRegr, GBTRegr, GPRegr, ZeroPred
from metrics import fast_bpr


def calc_score(model, x_df, y_df, metric):
    return calc_score_dict(model, x_df, y_df, '')[metric]

def calc_score_dict(model, x_df, y_df, split_name, timestep_col='timestep'):
    ytrue = y_df.values
    yhat = model.predict(x_df)

    mae = mean_absolute_error(ytrue, yhat)
    rmse = np.sqrt(mean_squared_error(ytrue, yhat))

    # BPR is calculated annually
    # get timesteps from x_df's index
    timesteps = x_df.index.get_level_values(timestep_col).unique()

    bpr_each_timestep = []
    for timestep in timesteps:
        ytrue_t = y_df[y_df.index.get_level_values(timestep_col) == timestep]
        yhat_t = yhat[x_df.index.get_level_values(timestep_col) == timestep]

        bpr_t = fast_bpr(pd.Series(np.squeeze(ytrue_t.values)), pd.Series(yhat_t), K=100)
        bpr_each_timestep.append(bpr_t)

    bpr = np.mean(bpr_each_timestep)

    return dict(
        mae=mae,
        rmse=rmse,
        bpr=bpr,
        neg_mae=-1.0*mae,
        neg_rmse=-1.0*rmse,
        max_yhat=np.max(yhat),
        method=model.__name__,
        hypers=str(model.get_params()),
        split=split_name)

def calc_score_dict_uncertainty(model, x_df, y_df, split_name,
                                timestep_col='timestep', geography_col='geoid',
                                uncertainty_samples=100, K=100,
                                seed=360, removed_locations=250):
    ytrue = y_df.values
    yhat = model.predict(x_df)

    # BPR is calculated annually
    # get timesteps from x_df's index
    timesteps = x_df.index.get_level_values(timestep_col).unique()

    rng = np.random.default_rng(seed=seed)

    locations = x_df.index.get_level_values(geography_col).unique()
    num_locations = len(locations)
    num_sampled_locations = num_locations - removed_locations

    mae_each_timestep =[]
    rmse_each_timestep = []
    bpr_each_timestep = []
    denominator_deaths_each_timestep = []
    deaths_reached_each_timestep = []
    for timestep in timesteps:
        ytrue_t = y_df[y_df.index.get_level_values('timestep') == timestep]
        yhat_t = yhat[x_df.index.get_level_values('timestep') == timestep]

        mae_each_sample =[]
        rmse_each_sample = []
        bpr_each_sample = []
        denominator_deaths_each_sample = []
        deaths_reached_each_sample = []
        
        for _ in range(uncertainty_samples):
            sampled_indicies = rng.choice(range(num_locations), size=num_sampled_locations, replace=False)

            ytrue_t_sampled = ytrue_t.iloc[sampled_indicies]
            yhat_t_sampled = yhat_t[sampled_indicies]

            denominator_deaths_t_sample = pd.Series(np.squeeze(ytrue_t_sampled.values)).sort_values().iloc[-K:].sum()

            mae_t_sampled = mean_absolute_error(ytrue_t_sampled, yhat_t_sampled)
            rmse_t_sampled = np.sqrt(mean_squared_error(ytrue_t_sampled, yhat_t_sampled))
            bpr_t_sampled = fast_bpr(pd.Series(np.squeeze(ytrue_t_sampled.values)), pd.Series(yhat_t_sampled), K=K)

            mae_each_sample.append(mae_t_sampled)
            rmse_each_sample.append(rmse_t_sampled)
            bpr_each_sample.append(bpr_t_sampled)
            denominator_deaths_each_sample.append(denominator_deaths_t_sample)
            deaths_reached_each_sample.append(bpr_t_sampled * denominator_deaths_t_sample)

        mae_each_timestep.append(mae_each_sample)
        rmse_each_timestep.append(rmse_each_sample)
        bpr_each_timestep.append(bpr_each_sample)
        denominator_deaths_each_timestep.append(denominator_deaths_each_sample)
        deaths_reached_each_timestep.append(deaths_reached_each_sample)

    
    mae_mean = np.mean(np.ravel(mae_each_timestep))
    rmse_mean = np.mean(np.ravel(rmse_each_timestep))
    bpr_mean = np.mean(np.ravel(bpr_each_timestep))
    deaths_reached_mean = np.mean(np.ravel(deaths_reached_each_timestep))

    mae_lower = np.percentile(np.ravel(mae_each_timestep), 2.5)
    rmse_lower = np.percentile(np.ravel(rmse_each_timestep), 2.5)
    bpr_lower = np.percentile(np.ravel(bpr_each_timestep), 2.5)
    deaths_reached_lower = np.percentile(np.ravel(deaths_reached_each_timestep), 2.5)

    mae_upper = np.percentile(np.ravel(mae_each_timestep), 97.5)
    rmse_upper = np.percentile(np.ravel(rmse_each_timestep), 97.5)
    bpr_upper = np.percentile(np.ravel(bpr_each_timestep), 97.5)
    deaths_reached_upper = np.percentile(np.ravel(deaths_reached_each_timestep), 97.5)

    return dict(
        mae_mean=mae_mean,
        mae_lower=mae_lower,
        mae_upper=mae_upper,
        rmse_mean=rmse_mean,
        rmse_lower=rmse_lower,
        rmse_upper=rmse_upper,
        bpr_mean=bpr_mean,
        bpr_lower=bpr_lower,
        bpr_upper=bpr_upper,
        deaths_reached_mean=deaths_reached_mean,
        deaths_reached_lower=deaths_reached_lower,
        deaths_reached_upper=deaths_reached_upper,
        max_yhat=np.max(yhat),
        method=model.__name__,
        hypers=str(model.get_params()),
        split=split_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--location', choices=['MA', 'cook'], help='Which location to run')
    # Argument to add models to list of models to be run
    parser.add_argument('--models', nargs='+', choices=['LastYear', 'LastWYears_Average', 'PoissonRegr', 'LinearRegr', 'GBTRegr', 'GPRegr', 'ZeroPred'],
                         help='Which models to run')
    # directory to save results in
    parser.add_argument('--results_dir', default='./best_models_stSVI', help='Directory to save results in')
    # args for space, time and svi
    parser.add_argument('--add_space', action='store_true', help='Whether to add space')
    parser.add_argument('--add_time', action='store_true', help='Whether to add time')
    parser.add_argument('--add_svi', action='store_true', help='Whether to add svi')
    parser.add_argument('--disp_names', nargs='*', help='Optional model display names')
    parser.add_argument('--train_start_year', type=int, help='Optional, year to start training data')
    # args for context
    parser.add_argument('--context_size_in_tsteps', type=int, default=10, help='How many timesteps of context to use')
    args = parser.parse_args()

    # convert args.models to list of model classes using the __name__ attribute
    models = [getattr(pipelines, model_name) for model_name in args.models]
    if args.disp_names is None:
        disp_names = [model.__name__ for model in models]
    else:
        disp_names = args.disp_names


    verbose = False

    context_size_in_tsteps = args.context_size_in_tsteps
    timestep_col = 'timestep'

    if args.location == 'MA':
        timescale = 'annual'
        csv_pattern_str = 'clean_{timescale}_tract'
        min_start_year = 2005

        if args.train_start_year is not None:
            train_start_year = max(args.train_start_year, min_start_year)
        else:
            train_start_year = min_start_year

        train_years= range(train_start_year, 2018+1) # GP might need later years
        valid_years=[2019]
        test_years=[2020, 2021]
        svi_cols = ['theme_1_pc', 'theme_2_pc', 'theme_3_pc',
                     'theme_4_pc', 'svi_pctile']
        space_cols =  ['lat', 'lon']

    elif args.location =='cook':
        timescale ='year'
        csv_pattern_str = 'cook_county_gdf_cleanwithsvi_{timescale}'

        min_start_year = 2016
        if args.train_start_year is not None:
            train_start_year = max(args.train_start_year, min_start_year)
        else:
            train_start_year = min_start_year

        train_years= range(train_start_year, 2019+1) # GP might need later years
        valid_years=[2020]
        test_years=[2021, 2022]
        svi_cols = [
            'svi_theme1', 'svi_theme2', 'svi_theme3',
            'svi_theme4', 'svi_total_']

        space_cols = [
            'INTPTLAT', 'INTPTLON']

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
        timestep_col=timestep_col,
        add_space=args.add_space,
        add_time=args.add_time,
        add_svi=args.add_svi)
    
    added_cols = []
    if args.add_space:
        added_cols += space_cols
    if args.add_time:
        added_cols += [timestep_col]
    if args.add_svi:
        added_cols += svi_cols
    
    for model_name, model_module in zip(disp_names, models):

        hyper_grid = model_module.make_hyper_grid(
            Wmax=context_size_in_tsteps, added_cols=added_cols)
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

            score = calc_score(model, va.x, va.y, 'bpr')

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

        best_model_dir = os.path.dirname(os.path.join(args.results_dir, args.location))
        best_model_path = os.path.join(best_model_dir, f"{model_name}_hyperparams.json")
        best_result_path = os.path.join(best_model_dir, f"{model_name}_results.csv")
        
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        best_model.save_params(best_model_path)

        # Record perf on val and test
        va_score_dict = calc_score_dict_uncertainty(best_model, va.x, va.y, 'valid')
        te_score_dict = calc_score_dict_uncertainty(best_model, te.x, te.y, 'test')
        df = pd.DataFrame([va_score_dict, te_score_dict]).copy()
        df.to_csv(best_result_path, index=False)

        paper_result_strings = f"{te_score_dict['bpr_mean']*100:.1f}, ({te_score_dict['bpr_lower']*100:.1f}- {te_score_dict['bpr_upper']*100:.1f})    " \
                               f"{te_score_dict['deaths_reached_mean']:.1f}    " \
                               f"{te_score_dict['mae_mean']:.2f}, ({te_score_dict['mae_lower']:.2f}- {te_score_dict['mae_upper']:.2f})    " \
                               f"{te_score_dict['rmse_mean']:.2f}, ({te_score_dict['rmse_lower']:.2f}- {te_score_dict['rmse_upper']:.2f})    "
        print(model.__name__)
        print(paper_result_strings)
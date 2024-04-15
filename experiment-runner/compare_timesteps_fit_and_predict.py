import argparse
import numpy as np
import pandas as pd
import os
import itertools
import make_xy_data_splits
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pipelines
from pipelines import LastYear, LastWYears_Average, PoissonRegr, LinearRegr, GBTRegr, GPRegr, ZeroPred
from metrics import fast_bpr

def calc_score(model, x_df, y_df, metric, x_fine_df = None, y_fine_df = None, c=-1):
    if c == -1:
        return calc_score_dict(model, x_df, y_df, '')[metric]
    
    return calc_split_score_dict(model, x_df, y_df, x_fine_df=x_fine_df, 
                                 y_fine_df=y_fine_df, split_name='', c=c)[metric]

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


def calc_split_score_dict(model, x_coarse_df, y_coarse_df, x_fine_df, y_fine_df, split_name, timestep_col='timestep', c=-1):

    yhat = model.predict(x_coarse_df)
    yhat_df = pd.DataFrame({'y_death_pred': yhat})
    yhat_df.index = y_coarse_df.index
    year_full_concat = pd.concat([y_coarse_df, yhat_df], axis=1)
    year_full_concat_updated = pd.DataFrame(columns = ['y_death_pred'])
    #get coarse timestep table ready to 

    for index, row in year_full_concat.iterrows():
        #expand each coarse row in the dataframe into c new rows by timestep
        #example: yearly df, each single row will become 4 new rows for a quarterly df
        row['timestep'] = index[1] * c - (c - 1)
        row['y_death_pred'] = row['y_death_pred'] / c
        row['geoid'] = index[0]
        year_full_concat_updated = pd.concat([year_full_concat_updated, pd.DataFrame([row])], ignore_index=True) 
        for j in range(1, c):
            new_row = row.copy() 
            new_row['timestep'] = index[1] * c + j - (c - 1)
            new_row['geoid'] = index[0]
            year_full_concat_updated = pd.concat([year_full_concat_updated, pd.DataFrame([new_row])], ignore_index=True)  

    year_full_concat_updated.set_index(['geoid', 'timestep'], inplace=True)

    mae = mean_absolute_error(year_full_concat_updated['deaths'].values, year_full_concat_updated['y_death_pred'].values)
    rmse = np.sqrt(mean_squared_error(year_full_concat_updated['deaths'].values, year_full_concat_updated['y_death_pred'].values))
   
    # BPR is calculated annually
    # get timesteps from x_df's index
    timesteps = x_fine_df.index.get_level_values(timestep_col).unique()

    bpr_each_timestep = []
    for timestep in timesteps:
        ytrue_t = y_fine_df[y_fine_df.index.get_level_values(timestep_col) == timestep]
        yhat_t = year_full_concat_updated[year_full_concat_updated.index.get_level_values(timestep_col) == timestep]['y_death_pred']
        bpr_t = fast_bpr(pd.Series(np.squeeze(ytrue_t.values)), pd.Series(np.squeeze(yhat_t.values)), K=100)
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

def calc_split_score_dict_uncertainty(model, x_coarse_df, y_coarse_df, x_fine_df, y_fine_df, 
                                      split_name, c,
                                timestep_col='timestep', geography_col='geoid',
                                uncertainty_samples=100, K=100,
                                seed=360, removed_locations=250):
    
    yhat = model.predict(x_coarse_df)
    yhat_df = pd.DataFrame({'y_death_pred': yhat})
    yhat_df.index = y_coarse_df.index #get coarse df ready for transform

    full_concat_coarse = pd.concat([y_coarse_df, yhat_df], axis=1)
    full_concat_fine = pd.DataFrame(columns = ['y_death_pred'])
    
    for index, row in full_concat_coarse.iterrows():
        #same as calc_split_score_dict: expand each row into c new rows
        row['timestep'] = index[1] * c - (c - 1)
        row['y_death_pred'] = row['y_death_pred'] / c
        row['geoid'] = index[0]
        row['deaths'] = row['deaths']
        full_concat_fine = pd.concat([full_concat_fine, pd.DataFrame([row])]) 
        for j in range(1, c):
            new_row = row.copy() 
            new_row['timestep'] = index[1] * c + j - (c - 1)
            new_row['geoid'] = index[0]
            full_concat_fine = pd.concat([full_concat_fine, pd.DataFrame([new_row])])  

    full_concat_fine.set_index(['geoid', 'timestep'], inplace=True)
    #full_concat_fine.to_csv('val_testing/geoid_timestep_coarsePred.csv')
    # BPR is calculated annually
    # get timesteps from x_df's index
    timesteps = x_fine_df.index.get_level_values(timestep_col).unique()

    rng = np.random.default_rng(seed=seed)

    locations = x_fine_df.index.get_level_values(geography_col).unique()
    num_locations = len(locations)
    num_sampled_locations = num_locations - removed_locations

    mae_each_timestep =[]
    rmse_each_timestep = []
    bpr_each_timestep = []
    denominator_deaths_each_timestep = []
    deaths_reached_each_timestep = []
    
    for timestep in timesteps:
       
        ytrue_t = y_fine_df[y_fine_df.index.get_level_values(timestep_col) == timestep]
        yhat_t = full_concat_fine[full_concat_fine.index.get_level_values(timestep_col) == timestep]['y_death_pred']
        
        mae_each_sample =[]
        rmse_each_sample = []
        bpr_each_sample = []
        denominator_deaths_each_sample = []
        deaths_reached_each_sample = []
        
        for _ in range(uncertainty_samples):
            sampled_indicies = rng.choice(range(num_locations), size=num_sampled_locations, replace=False)

            ytrue_t_sampled = ytrue_t.iloc[sampled_indicies]
            yhat_t_sampled = yhat_t.iloc[sampled_indicies]

            denominator_deaths_t_sample = pd.Series(np.squeeze(ytrue_t_sampled.values)).sort_values().iloc[-K:].sum()

            mae_t_sampled = mean_absolute_error(ytrue_t_sampled, yhat_t_sampled)
            rmse_t_sampled = np.sqrt(mean_squared_error(ytrue_t_sampled, yhat_t_sampled))
            bpr_t_sampled = fast_bpr(pd.Series(np.squeeze(ytrue_t_sampled.values)), pd.Series(np.squeeze(yhat_t_sampled.values)), K=K)

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
    #new additions: sammy
    parser.add_argument('--metric_to_fit', choices=['bpr', 'mae', 'rmse'], default = 'bpr', help='Metric by which model is fitted')
    parser.add_argument('--data_dir_path', default = '../cook-county/cleaning-cook-county/data_dir', help='path to data directory on local device')
    parser.add_argument('--longer_tstep', default='year')
    parser.add_argument('--shorter_tstep', default='quarterly')
    parser.add_argument('--tstep_ratio', default=1)
    
    args = parser.parse_args()
    #set timestep ratio
    c = int(args.tstep_ratio)

    if args.data_dir_path is None or not os.path.exists(args.data_dir_path):
        raise ValueError("Please set correct DATA_DIR on command line: --data_dir_path 'data_dir'")

    # convert args.models to list of model classes using the __name__ attribute
    models = [getattr(pipelines, model_name) for model_name in args.models]
    if args.disp_names is None:
        disp_names = [model.__name__ for model in models]
    else:
        disp_names = args.disp_names


    verbose = False

    context_size_in_tsteps = int(args.context_size_in_tsteps * c)
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

    elif args.location == 'cook':
        
        #STEP 1: run the study on the shorter tstep and print the results
        
        timescale = args.shorter_tstep
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
        data_dir = args.data_dir_path,
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

            score = calc_score(model, va.x, va.y, args.metric_to_fit)

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
        print(f'regular {args.shorter_tstep} metrics')
        print(model.__name__)
        print(paper_result_strings)

    #STEP 2: run model with baseline

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

    elif args.location == 'cook':
        
        #STEP 1: run the study on the shorter tstep and print the results
        
        timescale_coarse = args.longer_tstep
        csv_pattern_str = f'cook_county_gdf_cleanwithsvi_{timescale_coarse}'
    
    tr_coarse, va_coarse, te_coarse = make_xy_data_splits.load_xy_splits(
        data_dir = args.data_dir_path,
        timescale=timescale_coarse,
        csv_pattern_str=csv_pattern_str,
        train_years=train_years,
        valid_years=valid_years,
        test_years=test_years,
        context_size_in_tsteps=int(context_size_in_tsteps / c),
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
            Wmax=int(context_size_in_tsteps / c), added_cols=added_cols)
        keys = hyper_grid.keys()
        vals = hyper_grid.values()
        row_dict_list = list()

        best_score = -np.inf
        best_hypers = None
        best_model = None
        for hyper_vals in itertools.product(*vals):
            hypers = dict(zip(keys, hyper_vals))
            model = model_module.construct(**hypers)
            model.fit(tr_coarse.x, tr_coarse.y)

            score = calc_score(model, va_coarse.x, va_coarse.y, args.metric_to_fit, va.x, va.y, c=c)

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
        va_score_dict = calc_split_score_dict_uncertainty(best_model, va_coarse.x, va_coarse.y, va.x, va.y, 'valid', c=c)
        te_score_dict = calc_split_score_dict_uncertainty(best_model, te_coarse.x, te_coarse.y, te.x, te.y, 'test', c=c)

        paper_result_strings = f"{te_score_dict['bpr_mean']*100:.1f}, ({te_score_dict['bpr_lower']*100:.1f}- {te_score_dict['bpr_upper']*100:.1f})    " \
                            f"{te_score_dict['deaths_reached_mean']:.1f}    " \
                            f"{te_score_dict['mae_mean']:.2f}, ({te_score_dict['mae_lower']:.2f}- {te_score_dict['mae_upper']:.2f})    " \
                            f"{te_score_dict['rmse_mean']:.2f}, ({te_score_dict['rmse_lower']:.2f}- {te_score_dict['rmse_upper']:.2f})    "
        print('longer tstep baseline metrics')
        print(model.__name__)
        print(paper_result_strings)
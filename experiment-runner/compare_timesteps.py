import argparse
import os
import subprocess
import pandas as pd
import sys
import numpy as np
import copy

def run_tstep_df_creation(tstep, data_dir):
    if not os.path.exists(data_dir + f'/cook_county_gdf_cleanwithsvi_{tstep}.csv'):
        try:
            tstep_num = int(tstep)
            command = f'python ../cook-county/cleaning-cook-county/cook_county_total_script.py --custom_timestep_range {tstep} --data_dir {data_dir}'
            subprocess.run(command, shell=True)
        except ValueError:
            command = f'python ../cook-county/cleaning-cook-county/cook_county_total_script.py --str_tsteps {tstep} --data_dir {data_dir}'
            subprocess.run(command, shell=True)

    else:
        print(f'path exists to tstep {tstep}')

def assert_tsteps_line_up(longer_tstep, shorter_tstep, data_dir):
    '''
    Make sure the tsteps can be compared
    '''
    run_tstep_df_creation(longer_tstep, data_dir)
    run_tstep_df_creation(shorter_tstep, data_dir)

    longer_tstep_df = pd.read_csv(data_dir + f'/cook_county_gdf_cleanwithsvi_{longer_tstep}.csv')
    shorter_tstep_df = pd.read_csv(data_dir + f'/cook_county_gdf_cleanwithsvi_{shorter_tstep}.csv')

    #assert that the timesteps line up
    timesteps_lineup = max(shorter_tstep_df['timestep']) % max(longer_tstep_df['timestep']) == 0
    try:
        assert(timesteps_lineup)
        print('shorter timestep divides longer timestep')
    except AssertionError:
        print('timesteps cannot be compared, they do not line up')
        sys.exit(1)

    #assert that the death counts are the same
    tstep_ratio = int(max(shorter_tstep_df['timestep']) / max(longer_tstep_df['timestep']))

    shorter_tstep_deaths = np.array(shorter_tstep_df['deaths'])
    stretched_long_tstep = shorter_tstep_deaths.reshape((-1, tstep_ratio))

    sum_array = np.sum(stretched_long_tstep, axis=1)

    #assert deaths line up
    try:
        assert(np.array_equal(np.array(longer_tstep_df['deaths']), sum_array))
        print('dataframe death tallys are consistent')
    except AssertionError:
        print('dataframe death tallys are not consistent')
        sys.exit(1)

    print('dataframes meet criteria, proceeding to model fitting')

    return tstep_ratio


if __name__=='__main__':


    parser = argparse.ArgumentParser(description='Recreate table rows for a given location')
    parser.add_argument('--location', choices=['MA', 'cook'], default='cook', help='Location to recreate table rows for')
    parser.add_argument('--metric_to_fit', default= 'bpr', choices=['bpr', 'mae', 'rmse'], help='Metric by which to choose best model')
    parser.add_argument('--data_dir_path', default = '../cook-county/cleaning-cook-county/data_dir', help='path to data directory on local device')
    parser.add_argument('--rows', nargs='+', type=int, help='List of table rows to recreate')
    parser.add_argument('--longer_tstep', default = 'year')
    parser.add_argument('--shorter_tstep', default = 'quarterly')

    args = parser.parse_args()
    longer_tstep = args.longer_tstep
    shorter_tstep = args.shorter_tstep

    #asserts that deaths line up
    #returns 
    tstep_ratio = assert_tsteps_line_up(args.longer_tstep, args.shorter_tstep, args.data_dir_path)

    location = args.location
    rows = args.rows

    row_model_dict = {0: "ZeroPred",
                      1: "LastYear",
                      2: "LastWYears_Average",
                      3: "LinearRegr",
                      4: "PoissonRegr",
                      5: "PoissonRegr",
                      6: "GBTRegr",
                      7: "GPRegr"}
    
    row_name_dict = {0: "ZeroPred",
                      1: "LastYear",
                      2: "LastWYears_Average",
                      3: "LinearRegr",
                      4: "PoissonRegr",
                      5: "PoissonRegrSVI",
                      6: "GBTRegr",
                      7: "GPRegr"}
    
    row_add_space_time_svi_dict = {0: [False, False, False],
                                   1: [False, False, False],
                                   2: [False, False, False],
                                   3: [False, False, False],
                                   4: [True, True, False],
                                   5: [True, True, True],
                                   6: [True, True, True],
                                   7: [True, True, False]}
    row_Wmax_dict = {}
    row_Wmax_dict['MA'] = {0: 1,
                     1: 1,
                     2: 10,
                     3: 10,
                     4: 10,
                     5: 10,
                     6: 10,
                     7: 1,}
    
    row_Wmax_dict['cook'] = {0: 1,
                     1: 1,
                     2: 12,
                     3: 12,
                     4: 12,
                     5: 12,
                     6: 5,
                     7: 1,}
    
    # import and create a default dict that defaults to none for most rows
    # create dictionary of extra arguments for each row
    from collections import defaultdict
    row_extra_args_dict = defaultdict(lambda: None)
    row_extra_args_dict[7] = ['--train_start_year', '2015']
    metric_to_fit = args.metric_to_fit
    data_dir_path = args.data_dir_path

    for row in rows:

        model = row_model_dict[row]
        model_name = row_name_dict[row]
        add_space, add_time, add_svi = row_add_space_time_svi_dict[row]
        context_size_in_tsteps = row_Wmax_dict[location][row]

        results_dir = f'./best_models_{location}'
      
        command = f'python compare_timesteps_fit_and_predict.py --location {location} --models {model} --disp_names {model_name} --context_size_in_tsteps {context_size_in_tsteps} --metric_to_fit {metric_to_fit} --data_dir_path {data_dir_path} --longer_tstep {longer_tstep} --shorter_tstep {shorter_tstep} --tstep_ratio {tstep_ratio}'
        
        if add_space and add_time:
            results_dir += '_st'
            command += ' --add_space --add_time'
        if add_svi:
            results_dir +='SVI'
            command += ' --add_svi'

        extra_args = row_extra_args_dict[row]
        if extra_args is not None:
            for arg in extra_args:
                command += f' {arg}'

        command += f' --results_dir {results_dir}'
        
        print(command)
        # execute command in shell
        import subprocess
        subprocess.run(command, shell=True)
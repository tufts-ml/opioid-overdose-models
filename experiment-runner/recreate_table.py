import argparse



if __name__=='__main__':


    parser = argparse.ArgumentParser(description='Recreate table rows for a given location')
    parser.add_argument('--location', choices=['MA', 'cook'], help='Location to recreate table rows for')
    parser.add_argument('--rows', nargs='+', type=int, help='List of table rows to recreate')

    args = parser.parse_args()

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
                     2: 5,
                     3: 5,
                     4: 5,
                     5: 5,
                     6: 5,
                     7: 1,}
    
    # import and create a default dict that defaults to none for most rows
    # create dictionary of extra arguments for each row
    from collections import defaultdict
    row_extra_args_dict = defaultdict(lambda: None)
    row_extra_args_dict[7] = ['--train_start_year', '2015']


    for row in rows:

        model = row_model_dict[row]
        model_name = row_name_dict[row]
        add_space, add_time, add_svi = row_add_space_time_svi_dict[row]
        context_size_in_tsteps = row_Wmax_dict[location][row]

        results_dir = f'./best_models_{location}'
      
        command = f'python fit_and_predict.py --location {location} --models {model} --disp_names {model_name} --context_size_in_tsteps {context_size_in_tsteps}'
        
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
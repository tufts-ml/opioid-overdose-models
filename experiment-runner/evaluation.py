
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt



def calculate_metrics(evaluation_deaths, predicted_deaths, 
                      num_locations_removed = 250, confidence_level=0.95, seed=360,
                      num_uncertainty_samples=50):
        """
        @Return: joint RMSE and joint MAE alongside confidence interval
        @param: evaluation_deaths pulled from multiindexed_gdf, not sampled yet
        @param: predicted_deaths - corresponding model returns, already sampled
        """
        rng = np.random.default_rng(seed=seed) 
        num_years = len(evaluation_deaths)


        #make sure each element in evaluation_deaths is of same len, set num_locations to that len
        lengths = [len(sub_list) for sub_list in evaluation_deaths]
        if all(length == lengths[0] for length in lengths):
             num_locations = lengths[0] #1328 for cook county, 1620 for MA
        
        num_sampled = num_locations - num_locations_removed 

        #initialize lists to store values 
        mae_over_samples = [] 
        rmse_over_samples = []
        #calculate metrics for each year across diff. samples of predicted values and actual values
        for i in range(num_years): 
            for _ in range(num_uncertainty_samples):

                sampled_indices = rng.choice(range(num_locations), size=num_sampled, replace=False)
                current_eval_deaths = evaluation_deaths[i][sampled_indices]
                current_predicted_deaths = predicted_deaths[i][sampled_indices]

                mae_over_samples.append(mean_absolute_error(current_eval_deaths, current_predicted_deaths))
                rmse_over_samples.append(sqrt(mean_squared_error(current_eval_deaths, current_predicted_deaths)))

        #calculate mean and confidence interval (95%) based off joint rmse/mae vals
        joint_rmse_mean = np.mean(rmse_over_samples)
        joint_mae_mean = np.mean(mae_over_samples)
   
        #calculate mean and confidence interval (95%) based off joint rmse/mae vals
        confidence_level = max(0, min(confidence_level, 1)) 
        
        joint_rmse_lower = np.percentile(rmse_over_samples, (1 - confidence_level) * 100 / 2)
        joint_rmse_upper = np.percentile(rmse_over_samples, 100 - (1 - confidence_level) * 100 / 2)

        joint_mae_lower = np.percentile(mae_over_samples, (1 - confidence_level) * 100 / 2)
        joint_mae_upper = np.percentile(mae_over_samples, 100 - (1 - confidence_level) * 100 / 2)

        return (joint_rmse_mean, (joint_rmse_lower, joint_rmse_upper)), \
            (joint_mae_mean, (joint_mae_lower, joint_mae_upper))



###HELPER function to print results
def print_results(metric_name, mean_value, confidence_interval, confidence_level=0.95):
    '''Prints results from calculate_metrics'''
    print(f"{metric_name} (Mean, {confidence_level*100:.0f}% CI): {mean_value:.2f}, "
          f"({confidence_interval[0]:.2f}-{confidence_interval[1]:.2f})")

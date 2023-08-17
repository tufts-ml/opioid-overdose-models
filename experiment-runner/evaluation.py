
import numpy as np
import scipy.stats as stats


def calculate_metrics(actual_values, predicted_deaths, first_test_timestep, last_test_timestep, num_uncertainty_samples, 
                      confidence_level=0.95):
        """
        @Return: joint RMSE and joint MAE alongside confidence interval
        @param: num_uncertainty_samples should be SAME as bpr_uncertainty for that model
        """

        num_years = last_test_timestep - first_test_timestep + 1
        num_samples = num_uncertainty_samples

        joint_rmse_values = []
        joint_mae_values = []

        for year_idx in range(num_years):
            year_actual_values = actual_values[year_idx]
            year_predicted_deaths = predicted_deaths[year_idx]

            year_rmse_values = []
            year_mae_values = []

            for sample_idx in range(num_samples):
                samples = year_predicted_deaths[sample_idx]

                rmse = np.sqrt(np.mean((samples - year_actual_values)**2))
                year_rmse_values.append(rmse)

                mae_samples = np.mean(np.abs(samples - year_actual_values))
                year_mae_values.append(mae_samples)

            joint_rmse_values.extend(year_rmse_values)
            joint_mae_values.extend(year_mae_values)

        joint_rmse_mean = np.mean(joint_rmse_values)
        joint_rmse_conf_interval = stats.t.interval(confidence_level, len(joint_rmse_values) - 1, loc=joint_rmse_mean,
                                                    scale=stats.sem(joint_rmse_values))

        joint_mae_mean = np.mean(joint_mae_values)
        joint_mae_conf_interval = stats.t.interval(confidence_level, len(joint_mae_values) - 1, loc=joint_mae_mean,
                                                scale=stats.sem(joint_mae_values))

        return (joint_rmse_mean, joint_rmse_conf_interval), (joint_mae_mean, joint_mae_conf_interval)


###HElPER function to print results
def print_results(metric_name, mean_value, confidence_interval, confidence_level=0.95):
    '''Prints results from calculate_metrics'''
    print(f"{metric_name} (Mean, {confidence_level*100:.0f}% CI): {mean_value:.2f}, "
          f"({confidence_interval[0]:.2f}-{confidence_interval[1]:.2f})")

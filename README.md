# Spatiotemporal Forecasting of Opioid-related Fatal Overdoses: Towards Best Practices for Modeling and Evaluation


## Data

### Cook County
Cook County Data  is located in the `cook-county` directory. Running `extract_dataset.py` will recreate the necessary files

### Massachusetts
Massachusetts data is unable to be shared, but our cleaning scripts are included in the `massachusetts` directory

## Models

### Simple Python Models
Most models are located in the `experiment-runner` directory, and running `recreate_table.py` will re-create the results from Table 1

### Timestep Comparisons
Running `compare_timesteps.py --location [cook, MA] -- rows [desired models] --longer_tstep [timestep] -- shorter_tstep [timestep]` will create a comparison between the longer timestep and the shorter timestep's metrics. Namely, this comparison will show whether the shorter timestep is more useful in prediction than the longer timestep. 
This script verifies that the data exists first, then runs the experiment. 

### Other models
CASTNet, the Bayesian spatiotemporal model, and the negative binomial regression are more complex to run, and reside in their own directories
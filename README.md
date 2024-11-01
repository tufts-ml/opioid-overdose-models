# Spatiotemporal Forecasting of Opioid-related Fatal Overdoses: Towards Best Practices for Modeling and Evaluation

This is code for the published paper:

> *Spatiotemporal Forecasting of Opioid-related Fatal Overdoses: Towards Best Practices for Modeling and Evaluation* <br />
> Kyle Heuton, Jyontika Kapoor, Shikhar Shrestha, Thomas J Stopka, Michael C. Hughes <br />
> American Journal of Epidemiology, 2024 <br />
> <https://doi.org/10.1093/aje/kwae343>

Please get in touch with Kyle Heuton if you have questions.

## Data

### Cook County
Cook County Data  is located in the `cook-county` directory. Running `extract_dataset.py` will recreate the necessary files

### Massachusetts
Massachusetts data is unable to be shared, but our cleaning scripts are included in the `massachusetts` directory

## Models

### Simple Python Models
Most models are located in the `experiment-runner` directory, and running `recreate_table.py` will re-create the results from Table 1


### Other models
CASTNet, the Bayesian spatiotemporal model, and the negative binomial regression are more complex to run, and reside in their own directories

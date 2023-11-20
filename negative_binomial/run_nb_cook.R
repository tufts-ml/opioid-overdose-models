##
## Load the Needed Libraries
##
library(lme4)
library(MASS) 
library(sf)
library(rgdal)
library(spdep)
 
data_path <- "cook_county_gdf_cleanwithsvi_year"
end_year <- 2022
shape_path <- "tl_2021_17_tract"

data <- read_sf(data_path)
shape_data <- read_sf(shape_path)

# make weights
wm_q <- poly2nb(shape_data, queen = TRUE)
# row-standardized weights matrix i.e. for each row, the weight should total 1
rswm_q = nb2listw(wm_q, style = "W", zero.policy = TRUE)

# get data unique years
years <- unique(data$year)

# make sure deaths are numeric
data$deaths <- as.numeric(data$deaths)

data$death_sp_lag <- NA
data$next_year_overdose_deaths <- NA

for (year in years) {

    # get this year's data
    data_year <- data[data$year == year,]

    # make weights
    wm_q <- poly2nb(data_year, queen = TRUE)
    # row-standardized weights matrix i.e. for each row, the weight should total 1
    rswm_q = nb2listw(wm_q, style = "W", zero.policy = TRUE)

    data_year$death_sp_lag = lag.listw(rswm_q, data_year$deaths)

    if (year != max(years)){
        data_year$next_year_overdose_deaths = data[data$year == year + 1,]$deaths
    }

    # replace the data for this year with the new data
    data[data$year == year,] <- data_year
}



##
## Prior to running analysis we calculated the county-level carrying capacity
## See the manuscript for specific details
##
data$carrying_capacity <- NA

for(geoid in unique(data$geoid)){

     ## Carrying Capacity is initially set to 5% of the county population in 2015
    baseline <- data$pop[data$geoid == geoid & data$year == 2015]*0.05 
    
    for (year in 2015:end_year) {
        ## next we substract the number of overdose deaths from the prior three years
        ## noting that for 2016 we only subtract the prior year
        ## and that for 2017 only the prior two years

        if(year == 2015){ 
            data$carrying_capacity[data$geoid == geoid &
                                   data$year == year] <- baseline
        }
        else if(year == 2016){
            data$carrying_capacity[data$geoid == geoid & data$year == year] <- baseline -
                data$deaths[data$geoid == geoid & data$year == year - 1]
        }
        else if(year == 2017){  
            data$carrying_capacity[data$geoid == geoid & data$year == year] <- baseline -
                data$deaths[data$geoid == geoid & data$year == year - 1] -
                data$deaths[data$geoid == geoid & data$year == year - 2]
        }
        else{
            data$carrying_capacity[data$geoid == geoid & data$year == year] <- baseline -
                data$deaths[data$geoid == geoid & data$year == year - 1] -
                data$deaths[data$geoid == geoid & data$year == year - 2] -
                data$deaths[data$geoid == geoid & data$year == year - 3]
        } 
    }
}

## we then limit the carrying capacity such that its minimum value is 50
data$carrying_capacity[which(data$carrying_capacity < 50)] <- 50 


##
## Next we define our regression equation
##
func <- next_year_overdose_deaths ~ (year|geoid) + ## random effect for county (FIPS) with a random slope for year
 offset(log(carrying_capacity)) + ## offset term for the log of the carrying capacity
svi_theme1 + svi_theme2 + svi_theme3 + svi_theme4 + svi_total_

## First we create the results table where analysis for this half of the country will be stored

results <- data[data$year >= 2018, c("geoid","year")] 
## First, we create the columns for the observed and predicted SynthOD deaths rates in 2013 (i.e. 201X)
results$observed <- NA
results$predicted <- NA
for (year in 2018:end_year){
    ##
    ## We begin by predicting overdose death rates for the year 2013 (i.e. 201X)
    ## We provide detailed code for this year and note that the analysis for remaining years is identical
    ## As noted in the manuscript, in order to predict overdose deaths from the year 201X
    ## We take predictor data from the years 2015 - 201(X-2) (paired with outcome data from 2016 - 201(X-1))
    ## We train the model on this dataset
    ## Then we take predictors for the year 201(X-1) and feed them into the model to generate predictions for 201X15
    ## These values are the total predicted number of overdose deaths so then we population adjust
    ## them to get
    ## our final predicted overdose death rates for each county
    ##
    ## First we need the data to train our model
    ## We select all data corresponding to years 2015 through 2016 (i.e. 2015 - 201(X-2))
    train_data <- data[data$year >= 2015 & data$year < year-1 ,] 

    ## Next we need our test/prediction data
    ## We select the data corresponding to year 2017 (i.e. 201(X-1))
    test_data <- data[data$year == year-1,] 

    ## Next we train the model on our data using the function specified
    ## We set nAGQ = 0 in order to simplify the optimization routine
    ## which dramatically improves runtime
    lme_model <- glmer.nb(func, train_data, nAGQ = 0) 



    ## After training the model we need to use the model in order to predict the death counts for 2013 (201X)
    ## We use the predict function, supplying the trained model, the test/prediction data
    ## Making sure to set type = "response" to get the approrpriate metric
    ## We store the predicted value in the test_data data frame
    test_data$prediction <- predict(lme_model,newdata=test_data, type = "response")
    ## Finally we need to store the results
    ## The following code handles this
    

    ## Then we create a loop to go through every county in the results and we fill in the observed andpredicted values
    ## Note that the model predicts the crude count of deaths so we need to adjust the value by the population size
    ## In order to get the death rates
    this_years_results = results[results$year == year,]
    for(i in 1:nrow(this_years_results)){

        ## Extract the county FIPS code -- this is used as a key to match up data appropriately
        geoid <- this_years_results$geoid[i]
        ## Extract the population size of the county for the given year
        population <- test_data$pop[test_data$geoid == geoid]

        ## Get both the observed and predicted death counts for the year 2013 (i.e. 201X)
        observed <- test_data$next_year_overdose_deaths[test_data$geoid == geoid]
        predicted <- test_data$prediction[test_data$geoid == geoid]

        ## We then create the rate by dividing the deaths counts by the population size and then
        ## by multiplying this value by 100,000 (thus we have a death rate as X per 100,000)
        ## These values are stored in the appropriate location in the results table
        this_years_results$observed[i] <- observed
        this_years_results$predicted[i] <- predicted

    } 
    results[results$year == year,] <- this_years_results
}

# drop geometry colu
results$geometry <- NULL

write.csv(results, file = "./negative_binom_results.csv")


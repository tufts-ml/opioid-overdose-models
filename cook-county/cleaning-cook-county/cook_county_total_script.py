import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sklearn
import sys
import math
from dateutil.parser import parse
import geopandas as gpd
from shapely.geometry import Point
from timestep_funcs import map_season_semi, map_season_q, map_month, map_biweek, map_week
from shapely import wkt
import copy
import argparse

#COOK-COUNTY-INTRO functions
#write a function to get nonparsable dates
def get_non_parsable_dates(d, column):
    non_parsable_dates = []
    for date_value in d[column]:
        try:
            pd.to_datetime(str(date_value))
        except ValueError:
            non_parsable_dates.append(date_value)
    return non_parsable_dates

#CLEANING: modifies dataframe
def set_timestep_range(timestep):
    '''
    Given timestep, return range of timesteps
    '''
    if timestep == 'year':
        return range(1, 2)
    elif timestep == 'semiannual':
        return [1, 2]
    elif timestep == 'quarter':
        return range(1, 5)
    elif timestep == 'month':
        return range(1, 13)
    elif timestep == 'biweek':
        return range(1, 28)
    elif timestep == 'week':
        return range(1, 54)
    else:
        return range(1, 1 + int(timestep))

#append zero death columns
def create_square_df(gdf, timestep):
    updated_rows = []
    years = range(2015, 2024)
    if timestep == 'year':
        for tract in gdf['geoid'].unique():
            existing_years = set(gdf[gdf['geoid'] == tract]['year'])
            missing_years = set(years) - existing_years
            
            for year in missing_years:
                new_row = {'geoid': tract, 
                        'year': year, 
                        'deaths': 0}
                
                updated_rows.append(new_row)
    else:   
        try:
            #for int based timesteps
            try_str_timestep = int(timestep)
            timesteps = set_timestep_range(timestep)
            existing_combinations = set(zip(gdf['geoid'], gdf['year'], gdf[timestep]))

            for tract in gdf['geoid'].unique():
                for year in years:
                    for time_ in timesteps:
                        combination = (tract, year, time_)
                        if combination not in existing_combinations:
                            new_row = {'geoid': tract, 'year': year, timestep: time_, 'deaths': 0}
                            updated_rows.append(new_row)
        except ValueError:
            #for preset string timesteps
            timesteps = set_timestep_range(timestep)
            existing_combinations = set(zip(gdf['geoid'], gdf['year'], gdf[timestep]))

            for tract in gdf['geoid'].unique():
                for year in years:
                    for time_ in timesteps:
                        combination = (tract, year, time_)
                        if combination not in existing_combinations:
                            new_row = {'geoid': tract, 'year': year, timestep: time_, 'deaths': 0}
                            updated_rows.append(new_row)
        
    return updated_rows

#reformat specific timestep strings
def reformat_timestep(timestep_col, timestep):
    timestep_col = timestep_col.astype(str)
    final_timestep = list(set_timestep_range(timestep))[-1]
    for i in range(1, final_timestep + 1):
        timestep_col = timestep_col.replace(str(i) + '.0', str(i))

    return timestep_col

def get_sort_by_col(timestep):
    if timestep == 'year':
        return ['geoid', 'year']
    else:
        return ['geoid', 'year', timestep]

def get_timestep_identifier_name(timestep):
    '''
    Returns how to call the timestep 
    '''
    if timestep == 'semiannual' or timestep == 'quarter':
        return 'season'
    elif timestep == 'month':
        return 'month_name'
    else:
        return f'{timestep}_id'

def add_aux_cols(gdf, timestep):
    '''
    Add higher order cols to do timescale analysis
    '''
    if timestep == 'biweekly':
        gdf['month'] = gdf['biweek'].astype(np.int32).map(lambda x: np.ceil(x / 2.25))
        gdf['quarter'] = gdf['biweek'].astype(np.int32).map(lambda x: 1 if x <= 7 else 2 if x <= 14 else 3 if x <= 21 else 4)
        gdf['semiannual'] = gdf['biweek'].astype(np.int32).map(lambda x: 1 if x <= 13 else 2)
    elif timestep == 'month':
        gdf['semiannual'] = gdf['month'].astype(np.int32).map(lambda x: 1 if x <= 6 else 2)
        gdf['quarter'] = gdf['month'].astype(np.int32).map(lambda x: 1 if x <= 3 else 2 if x <= 6 else 3 if x <= 9 else 4)
    elif timestep == 'quarter':
        gdf['semiannual'] = gdf['quarter'].astype(np.int32).map(lambda x: 1 if x <= 2 else 2)

# Create timstep_identifier column
def map_timestep(time, timestep):
    if timestep == 'year':
        return None
    elif timestep == 'semiannual':
        return map_season_semi(time)
    elif timestep == 'quarter':
        return map_season_q(time)
    elif timestep == 'month':
        return map_month(time)
    elif timestep == 'biweek':
        return map_biweek(time)
    elif timestep == 'week':
        return map_week(time)

def get_timestep_cols_to_keep(timestep):
    '''
    Returns the number of timesteps we want
    '''
    if timestep == 'semiannual':
        return ['semiannual', 'season']
    elif timestep == 'quarter':
        return ['semiannual', 'quarter', 'season']
    elif timestep == 'month':
        return ['semiannual', 'quarter', 'month', 'month_name']
    elif timestep == 'biweek':
        return ['biweek', 'biweek_id']
    else:
        return [str(timestep)]
    

def clean_gdf(cook_county_gdf, timestep):
    cook_county_gdf[timestep].fillna(9999, inplace=True)
    cook_county_gdf['year'].fillna(9999, inplace=True)
    columns_to_keep = [
    'INTPTLAT', 'INTPTLON', 
    'STATEFP', 'COUNTYFP', 'TRACTCE','NAME', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'ALAND', 'AWATER',
    'geometry']
    gdf = cook_county_gdf.groupby(get_sort_by_col(timestep)).agg({'deaths': 'sum', **{col: 'first' for col in columns_to_keep}}).reset_index()

    gdf.loc[gdf[timestep] == 9999, 'deaths'] = 0
    gdf.loc[gdf['year'] == 9999, 'deaths'] = 0
    gdf['year'].replace(9999, 2015, inplace=True)
    if timestep != 'year': gdf[timestep].replace(9999, 1, inplace=True)

    # Fill in missing tract-year-period cells
    # Warning: Nested for loops
    updated_rows = create_square_df(gdf, timestep)
    gdf = pd.concat([gdf, pd.DataFrame(updated_rows)], ignore_index=True)


    unique_tracts = gdf['geoid'].unique()
    for tract in unique_tracts:
        tract_rows = gdf[gdf['geoid'] == tract]
        non_na_row = tract_rows.dropna().iloc[0]  # Get the first row without any NAs
        
        for column in columns_to_keep:
            gdf.loc[(gdf['geoid'] == tract) & gdf[column].isna(), column] = non_na_row[column]

    #change format of year and quarters
                
    if timestep != 'year':
        
        gdf['year'] = pd.to_datetime(gdf['year'].astype(int).astype(str), format='%Y').dt.year
        
        gdf[timestep] = gdf[timestep].astype(str)
        final_timestep = list(set_timestep_range(timestep))[-1]
        for i in range(1, final_timestep + 1):
            gdf[timestep] = gdf[timestep].replace(str(i) + '.0', str(i))

        #take out 2023 and 2014
        gdf = gdf[gdf['year'] != 2023]
        gdf = gdf[gdf['year'] != 2014]
        
        # Create timestep column
        add_aux_cols(gdf, timestep)
        gdf[timestep] = gdf[timestep].astype(int)
        gdf = gdf.sort_values(by=get_sort_by_col(timestep)).reset_index(drop=True).copy()
        #add cols with longer timescales than the current timestep
        gdf['timestep'] = gdf.groupby(['geoid']).cumcount() + 1
        timestep_identifier_name = get_timestep_identifier_name(timestep) #could be season, month_name, etc
        gdf[timestep_identifier_name] = gdf[timestep].map(lambda x: map_timestep(x, timestep))


        final_timestep = list(set_timestep_range(timestep))[-1]
        gdf['year_frac'] = gdf['year'].values + (gdf[timestep].values.astype('int') - 1) / final_timestep
        gdf = gdf[[
        'geoid', 'timestep', 'year_frac', 
        'deaths',
        'year'] + get_timestep_cols_to_keep(timestep) + columns_to_keep].copy()
        #convert back to gdf
        gdf['geometry'] = gdf['geometry'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
        gdf.crs = {'init': 'EPSG:4269'}

    else:
        years = range(2015, 2023)  # data collect started in late '14, we ran this notebook in mid '23
        gdf_keep = copy.deepcopy(gdf).query("year >= %d & year <= %d" % (years[0], years[-1]))

        #Convert back to gdf
        gdf_keep['geometry'] = gdf_keep['geometry'].apply(wkt.loads)
        new_gdf = gpd.GeoDataFrame(gdf_keep, geometry='geometry')
        new_gdf.crs = {'init': 'EPSG:4269'}

        new_gdf = new_gdf.sort_values(by=['geoid', 'year']).reset_index(drop=True).copy()
        new_gdf['timestep'] = new_gdf.groupby(['geoid']).cumcount() + 1
        new_gdf['year_frac'] = new_gdf['year'].copy()
        new_gdf = new_gdf[[
        'geoid', 'timestep', 'year_frac', 
        'deaths',
        'year']
        + columns_to_keep
        ].copy()
        new_gdf['year'] = pd.to_datetime(new_gdf['year'].astype(int).astype(str), format='%Y').dt.year
        return new_gdf

    return gdf

def get_timestep_cols_SVI(timestep):
    '''
    Gets columns for specific SVI df
    '''
    base_cols = 'geoid,timestep,year_frac,deaths,year'
    svi_cols = 'svi_theme1_pctile,svi_theme2_pctile,svi_theme3_pctile,svi_theme4_pctile,svi_total_pctile,pop'
    geo_cols = 'INTPTLAT,INTPTLON,STATEFP,COUNTYFP,TRACTCE,NAME,NAMELSAD,MTFCC,FUNCSTAT,ALAND,AWATER,geometry'

    if timestep == 'year':
        return base_cols.split(',') + svi_cols.split(',') + geo_cols.split(',')
    else:   
        return base_cols.split(',')  + get_timestep_cols_to_keep(timestep) + svi_cols.split(',') + geo_cols.split(',')


def clean_with_svi(cook_county_gdf, timestep):
    '''
    Add SVI data
    '''

    #convert to gpd (was having trouble importing csv as gdf)
    cook_county_gdf['geometry'] = cook_county_gdf['geometry'].apply(wkt.loads)
    cook_county_gdf = gpd.GeoDataFrame(cook_county_gdf, geometry='geometry')
    cook_county_gdf.crs = {'init': 'EPSG:4269'}

    svi_2016_dir = os.path.join(data_dir, 'cook-county-svi-data', 'cook-county-svi-2016.csv')
    svi_2018_dir = os.path.join(data_dir, 'cook-county-svi-data', 'cook-county-svi-2018.csv')
    svi_2020_dir = os.path.join(data_dir, 'cook-county-svi-data', 'cook-county-svi-2020.csv')

    svi_2016 = pd.read_csv(svi_2016_dir)
    svi_2018 = pd.read_csv(svi_2018_dir)
    svi_2020 = pd.read_csv(svi_2020_dir)

    cook_county_gdf['geoid'] = cook_county_gdf['geoid'].astype(str)

    cook_county_gdf = cook_county_gdf.assign(
        RPL_THEME1 = float('nan'),
        RPL_THEME2 = float('nan'),
        RPL_THEME3 = float('nan'),
        RPL_THEME4 = float('nan'),
        RPL_THEMES = float('nan'),
        E_TOTPOP = float('nan'))

    cook_county_gdf = cook_county_gdf.reset_index(drop=True)

    #add in svi_2016 to cook_county_gdf
    svi_2016['FIPS'] = svi_2016['FIPS'].astype(str)
    cook_county_gdf['geoid'] = cook_county_gdf['geoid'].str.strip()
    years_to_update = list(range(2015, 2017))

    # Iterate over each row in cook_county_gdf where 2015 <= year <= 2016
    for index, row in cook_county_gdf[cook_county_gdf['year'].isin(years_to_update)].iterrows():
        geoid_value = row['geoid']
        matching_row = svi_2016[svi_2016['FIPS'] == geoid_value]
        
        # If a matching row is found, update the corresponding columns in cook_county_gdf
        if not matching_row.empty:
            cook_county_gdf.loc[index, 'RPL_THEME1'] = matching_row['RPL_THEME1'].values[0]
            cook_county_gdf.loc[index, 'RPL_THEME2'] = matching_row['RPL_THEME2'].values[0]
            cook_county_gdf.loc[index, 'RPL_THEME3'] = matching_row['RPL_THEME3'].values[0]
            cook_county_gdf.loc[index, 'RPL_THEME4'] = matching_row['RPL_THEME4'].values[0]
            cook_county_gdf.loc[index, 'RPL_THEMES'] = matching_row['RPL_THEMES'].values[0]
            cook_county_gdf.loc[index, 'E_TOTPOP'] = matching_row['E_TOTPOP'].values[0]

    #add in svi_2018 to cook_county_gdf
    svi_2018['FIPS'] = svi_2018['FIPS'].astype(str)
    cook_county_gdf['geoid'] = cook_county_gdf['geoid'].str.strip()
    years_to_update = list(range(2017, 2019))

    # Iterate over each row in cook_county_gdf where 2017 <= year <= 2018
    for index, row in cook_county_gdf[cook_county_gdf['year'].isin(years_to_update)].iterrows():
        geoid_value = row['geoid']
        matching_row = svi_2018[svi_2018['FIPS'] == geoid_value]
        
        # If a matching row is found, update the corresponding columns in cook_county_gdf
        if not matching_row.empty:
            cook_county_gdf.loc[index, 'RPL_THEME1'] = matching_row['RPL_THEME1'].values[0]
            cook_county_gdf.loc[index, 'RPL_THEME2'] = matching_row['RPL_THEME2'].values[0]
            cook_county_gdf.loc[index, 'RPL_THEME3'] = matching_row['RPL_THEME3'].values[0]
            cook_county_gdf.loc[index, 'RPL_THEME4'] = matching_row['RPL_THEME4'].values[0]
            cook_county_gdf.loc[index, 'RPL_THEMES'] = matching_row['RPL_THEMES'].values[0]
            cook_county_gdf.loc[index, 'E_TOTPOP'] = matching_row['E_TOTPOP'].values[0]

    #add in svi_2020 to cook_county_gdf
    svi_2020['FIPS'] = svi_2020['FIPS'].astype(str)
    cook_county_gdf['geoid'] = cook_county_gdf['geoid'].str.strip()
    years_to_update = list(range(2019, 2023))

    # Iterate over each row in cook_county_gdf where 2019 <= year <= 2022
    for index, row in cook_county_gdf[cook_county_gdf['year'].isin(years_to_update)].iterrows():
        geoid_value = row['geoid']
        matching_row = svi_2020[svi_2020['FIPS'] == geoid_value]
        
        # If a matching row is found, update the corresponding columns in cook_county_gdf
        if not matching_row.empty:
            cook_county_gdf.loc[index, 'RPL_THEME1'] = matching_row['RPL_THEME1'].values[0]
            cook_county_gdf.loc[index, 'RPL_THEME2'] = matching_row['RPL_THEME2'].values[0]
            cook_county_gdf.loc[index, 'RPL_THEME3'] = matching_row['RPL_THEME3'].values[0]
            cook_county_gdf.loc[index, 'RPL_THEME4'] = matching_row['RPL_THEME4'].values[0]
            cook_county_gdf.loc[index, 'RPL_THEMES'] = matching_row['RPL_THEMES'].values[0]
            cook_county_gdf.loc[index, 'E_TOTPOP'] = matching_row['E_TOTPOP'].values[0]

    #dropping census tracts with zero population (lake)
    geoid_to_drop = ['17031990000', '17031381700', '17031980000', '17031980100']
    cook_county_gdf = cook_county_gdf[~cook_county_gdf['geoid'].isin(geoid_to_drop)]

    #populate NAs for geoids that are only missing *some* rows

    columns_to_fill = ['RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4', 'RPL_THEMES', 'E_TOTPOP']

    # check for NA vals in each row and sum the number of NA values
    na_counts = cook_county_gdf.isna().sum(axis=1)
    # filter rows with NA values
    rows_with_na = cook_county_gdf[na_counts > 0]
    geoid_values_with_na = rows_with_na['geoid'].unique()
    for geoid_value in geoid_values_with_na:
        rows_for_geoid = cook_county_gdf[cook_county_gdf['geoid'] == geoid_value]
        rows_with_values = rows_for_geoid.dropna(subset=columns_to_fill, how='any')
        
        if not rows_with_values.empty:
            values_to_fill = rows_with_values.iloc[0][columns_to_fill].to_dict()
        
            cook_county_gdf.loc[cook_county_gdf['geoid'] == geoid_value, columns_to_fill] = cook_county_gdf.loc[
                cook_county_gdf['geoid'] == geoid_value, columns_to_fill
            ].fillna(values_to_fill)
    


    #rename columns to match MA
    old2new_cols_dict = {
        'RPL_THEME1': 'svi_theme1_pctile',
        'RPL_THEME2': 'svi_theme2_pctile',
        'RPL_THEME3': 'svi_theme3_pctile',
        'RPL_THEME4': 'svi_theme4_pctile',
        'RPL_THEMES': 'svi_total_pctile',
        'E_TOTPOP': 'pop'
    }

    cook_county_gdf = cook_county_gdf.replace(-999.0000, np.nan)
    cook_county_gdf.rename(columns=old2new_cols_dict, inplace=True)
    cook_county_gdf = cook_county_gdf[get_timestep_cols_SVI(timestep)].copy()

    return cook_county_gdf

if __name__ == '__main__':

    all_tsteps = 'year,semiannual,quarterly,monthly,biweekly,weekly'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir_path', default='data_dir', type=str)
    parser.add_argument('--custom_timestep_range', default=0, type=int)
    parser.add_argument('--str_tsteps', default = '', type=str)
    args = parser.parse_args()
    custom_timestep = args.custom_timestep_range
    print(custom_timestep)
    str_timesteps = args.str_tsteps.split(',')
    
    #set data directory on personal device
    data_dir = os.environ.get('DATA_DIR', args.data_dir_path)
    if data_dir is None or not os.path.exists(data_dir):
        raise ValueError("Please set correct DATA_DIR on command line: --data_dir_path 'data_dir'")

    csv_path = os.path.join(data_dir, 'cook-county-opioid.csv') #csv located on cluster
    d = pd.read_csv(csv_path)
    d["DEATH_DATE"] = pd.to_datetime(d["DEATH_DATE"])
    non_parsable_dates = get_non_parsable_dates(d, "INCIDENT_DATE")
    d = d[~d['INCIDENT_DATE'].isin(non_parsable_dates)].reset_index(drop=True)
    d["INCIDENT_DATE"] = pd.to_datetime(d["INCIDENT_DATE"])

    d['Time_Difference'] = d['DEATH_DATE'] - d['INCIDENT_DATE']

    # Load census tract shapefile
    shapefile_dir = os.path.join(data_dir, 'shapefiles')  #shapefiles are on cluster
    me_shape_path = os.path.join(shapefile_dir, 'ME_Cook/Medical_Examiner_Case_Archive%2C_2014_to_present.shp')
    tl_shape_path = os.path.join(shapefile_dir, 'tl_2021_17_tract/tl_2021_17_tract.shp')

    me_gdf = gpd.read_file(me_shape_path)
    tl_gdf = gpd.read_file(tl_shape_path)

    me_gdf.to_crs(tl_gdf.crs, inplace=True)

    me_gdf = me_gdf.dropna(subset=['geometry']) #drop people with missing Geometry values (lat, long)
        
    tl_gdf['deaths'] = 0
    missing_geo = 0
    for i, death in me_gdf.iterrows():
        
        if death.geometry is None:
            missing_geo +=1
            continue
        
        containing_tract_idx = tl_gdf.contains(death.geometry)
        assert(containing_tract_idx.sum()==1)
        tl_gdf.loc[containing_tract_idx, 'deaths'] += 1

    # spatial join between the deaths GeoDataFrame and the filtered cook_county_tracts
    cook_county_tracts = tl_gdf[tl_gdf['COUNTYFP'] == '031']
    cook_county_gdf = gpd.sjoin(cook_county_tracts, me_gdf, how='left', predicate='contains')
    cook_county_gdf.rename(columns={'GEOID': 'geoid'}, inplace=True)

    # Set deaths to 1, except where the merge failed
    cook_county_gdf.loc[cook_county_gdf['deaths']!=0, 'deaths'] = 1

    #make cols for each timestep
    cook_county_gdf.dropna(subset=['geoid'], inplace=True)
    cook_county_gdf["DEATH_DATE"] = pd.to_datetime(cook_county_gdf["DEATH_DATE"])
    cook_county_gdf['year'] = cook_county_gdf['DEATH_DATE'].dt.year #
    cook_county_gdf['quarter'] = cook_county_gdf['DEATH_DATE'].dt.quarter
    cook_county_gdf['semiannual'] = cook_county_gdf['DEATH_DATE'].dt.quarter.apply(lambda x: 1 if x <= 2 else 2)
    cook_county_gdf['month'] = cook_county_gdf['DEATH_DATE'].dt.month
    cook_county_gdf['biweek'] = cook_county_gdf['DEATH_DATE'].dt.isocalendar().week.apply(lambda x: math.ceil(x / 2) if not pd.isna(x) else x)
    cook_county_gdf['week'] = cook_county_gdf['DEATH_DATE'].dt.isocalendar().week

    if args.custom_timestep_range != 0.0: 
        cook_county_gdf[str(custom_timestep)] = cook_county_gdf.apply(lambda row: np.nan if pd.isna(row['DEATH_DATE']) 
                                                                                                  else math.ceil((row['DEATH_DATE'].dayofyear * custom_timestep) /  366) if row['year'] % 4 == 0 
                                                                                                  else math.ceil((row['DEATH_DATE'].dayofyear * custom_timestep) /  365), axis=1)
        
        cook_county_gdf_custom = cook_county_gdf.loc[:, [str(custom_timestep), 'year', 'STATEFP', 'COUNTYFP', 'TRACTCE',
            'geoid', 'NAME', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'ALAND', 'AWATER',
            'INTPTLAT', 'INTPTLON', 'geometry', 'deaths' ]]
    
        cook_county_gdf_custom.to_csv(f'{data_dir}/cook_county_gdf_intro_custom_{str(args.custom_timestep_range)}.csv', index=False)
        csv_path = os.path.join(data_dir, f'cook_county_gdf_intro_custom_{str(args.custom_timestep_range)}.csv')
        cook_county_gdf_custom = pd.read_csv(csv_path) 
        gdf_custom = clean_gdf(cook_county_gdf_custom, str(custom_timestep))
        gdf_custom.to_csv(csv_path.replace("intro", "clean"), index=False)
        gdf_custom_cleanwithsvi = clean_with_svi(pd.read_csv(csv_path.replace("intro", "clean")), 'year')
        gdf_custom_cleanwithsvi.to_csv(f'{data_dir}/cook_county_gdf_cleanwithsvi_custom_{str(args.custom_timestep_range)}.csv', index=False)
        gdf_custom_cleanwithsvi.to_file(f'{data_dir}/cook_county_gdf_cleanwithsvi_custom_{str(args.custom_timestep_range)}', index=False)
    
    for tstep in str_timesteps:
        tstep_dict = {'year': 'year', 'semiannual': 'semiannual', 'quarterly': 'quarter', 'monthly': 'month', 'biweekly': 'biweek', 'weekly': 'week'}

        if tstep == 'year':
            cook_county_gdf = cook_county_gdf.loc[:, ['year', 'STATEFP', 'COUNTYFP', 'TRACTCE',
        'geoid', 'NAME', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'ALAND', 'AWATER',
        'INTPTLAT', 'INTPTLON', 'geometry', 'deaths' ]]
        else:
            cook_county_gdf = cook_county_gdf.loc[:, [tstep_dict[tstep], 'year', 'STATEFP', 'COUNTYFP', 'TRACTCE',
        'geoid', 'NAME', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'ALAND', 'AWATER',
        'INTPTLAT', 'INTPTLON', 'geometry', 'deaths' ]]
        
        cook_county_gdf.to_csv(f'{data_dir}/cook_county_gdf_intro_{tstep}.csv', index=False)

        csv_path = os.path.join(data_dir, f'cook_county_gdf_intro_{tstep}.csv')
        cook_county_gdf = pd.read_csv(csv_path) 

        gdf = clean_gdf(cook_county_gdf, tstep_dict[tstep])

        gdf.to_csv(csv_path.replace("intro", "clean"), index=False)
        gdf_cleanwithsvi = clean_with_svi(pd.read_csv(csv_path.replace("intro", "clean")), tstep_dict[tstep])
        gdf_cleanwithsvi.to_csv(f'{data_dir}/cook_county_gdf_cleanwithsvi_{tstep}.csv', index=False)
        gdf_cleanwithsvi.to_file(f'{data_dir}/cook_county_gdf_cleanwithsvi_{tstep}', index=False)
    #subset columns that are relevant on non-individual level basis

    #This section creates the string-based dataframes
    #Currently has just year, semiannual, and quarterly to completion, but can easily add 
    #monthly, biweekly, and weekly

    cook_county_gdf_year = cook_county_gdf.loc[:, ['year', 'STATEFP', 'COUNTYFP', 'TRACTCE',
        'geoid', 'NAME', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'ALAND', 'AWATER',
        'INTPTLAT', 'INTPTLON', 'geometry', 'deaths' ]]

    cook_county_gdf_semiannual = cook_county_gdf.loc[:, ['semiannual','year', 'STATEFP', 'COUNTYFP', 'TRACTCE',
        'geoid', 'NAME', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'ALAND', 'AWATER',
        'INTPTLAT', 'INTPTLON', 'geometry', 'deaths' ]]
    
    cook_county_gdf_quarterly = cook_county_gdf.loc[:, ['quarter', 'year', 'STATEFP', 'COUNTYFP', 'TRACTCE',
        'geoid', 'NAME', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'ALAND', 'AWATER',
        'INTPTLAT', 'INTPTLON', 'geometry', 'deaths' ]]

    cook_county_gdf_monthly = cook_county_gdf.loc[:, ['month', 'year', 'STATEFP', 'COUNTYFP', 'TRACTCE',
        'geoid', 'NAME', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'ALAND', 'AWATER',
        'INTPTLAT', 'INTPTLON', 'geometry', 'deaths' ]]

    cook_county_gdf_biweekly = cook_county_gdf.loc[:, ['biweek', 'year', 'STATEFP', 'COUNTYFP', 'TRACTCE',
        'geoid', 'NAME', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'ALAND', 'AWATER',
        'INTPTLAT', 'INTPTLON', 'geometry', 'deaths' ]]

    cook_county_gdf_weekly = cook_county_gdf.loc[:, ['week', 'year', 'STATEFP', 'COUNTYFP', 'TRACTCE',
        'geoid', 'NAME', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'ALAND', 'AWATER',
        'INTPTLAT', 'INTPTLON', 'geometry', 'deaths' ]]

    cook_county_gdf_year.to_csv(f'{data_dir}/cook_county_gdf_intro_year.csv', index=False)
    cook_county_gdf_quarterly.to_csv(f'{data_dir}/cook_county_gdf_intro_quarterly.csv', index=False)
    cook_county_gdf_semiannual.to_csv(f'{data_dir}/cook_county_gdf_intro_semiannual.csv', index=False)

    csv_path_year = os.path.join(data_dir, 'cook_county_gdf_intro_year.csv')
    cook_county_gdf_year = pd.read_csv(csv_path_year) 
    csv_path_semi = os.path.join(data_dir, 'cook_county_gdf_intro_semiannual.csv')
    cook_county_gdf_semiannual = pd.read_csv(csv_path_semi) 
    csv_path_q = os.path.join(data_dir, 'cook_county_gdf_intro_quarterly.csv')
    cook_county_gdf_quarterly = pd.read_csv(csv_path_q) 

    gdf_annual = clean_gdf(cook_county_gdf_year, 'year')
    gdf_semiannual = clean_gdf(cook_county_gdf_semiannual, 'semiannual')
    gdf_quarter = clean_gdf(cook_county_gdf_quarterly, 'quarter')

    gdf_annual.to_csv(csv_path_year.replace("intro", "clean"), index=False)
    gdf_semiannual.to_csv(csv_path_semi.replace("intro", "clean"), index=False)
    gdf_quarter.to_csv(csv_path_q.replace("intro", "clean"), index=False)

    gdf_annual_cleanwithsvi = clean_with_svi(pd.read_csv(csv_path_year.replace("intro", "clean")), 'year')
    gdf_semiannual_cleanwithsvi = clean_with_svi(pd.read_csv(csv_path_semi.replace("intro", "clean")), 'semiannual')
    gdf_quarter_cleanwithsvi = clean_with_svi(pd.read_csv(csv_path_q.replace("intro", "clean")), 'quarter')

    gdf_annual_cleanwithsvi.to_csv(f'{data_dir}/cook_county_gdf_cleanwithsvi_year.csv', index=False)
    gdf_semiannual_cleanwithsvi.to_csv(f'{data_dir}/cook_county_gdf_cleanwithsvi_semiannual.csv', index=False)
    gdf_quarter_cleanwithsvi.to_csv(f'{data_dir}/cook_county_gdf_cleanwithsvi_quarterly.csv', index=False)

    gdf_annual_cleanwithsvi.to_file(f'{data_dir}/cook_county_gdf_cleanwithsvi_year', index=False)
    gdf_semiannual_cleanwithsvi.to_file(f'{data_dir}/cook_county_gdf_cleanwithsvi_semiannual', index=False)
    gdf_quarter_cleanwithsvi.to_file(f'{data_dir}/cook_county_gdf_cleanwithsvi_quarterly', index=False)
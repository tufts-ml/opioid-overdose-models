import os
import argparse
from glob import glob
import time

import numpy as np
import pandas as pd
idx = pd.IndexSlice
from censusgeocode import CensusGeocode
import geopandas as gpd

from math import radians, cos, sin, asin, sqrt

def get_available_files_and_years(data_dir):

    # Not every file is an xlsx, some just xls
    pre_2017_wildcard = os.path.join(data_dir,'MAOpiDths20*_clean.xls*')
    post_2017_wildcard = os.path.join(data_dir, 'RVRS_Opioids*')

    pre_2017_files = glob(pre_2017_wildcard)
    pre_2017_years = [int(os.path.basename(file).split('_')[0][-4:]) for file in pre_2017_files]

    post_2017_files = glob(post_2017_wildcard)
    post_2017_years = [int(os.path.splitext(os.path.basename(file))[0].split('_')[2]) for file in post_2017_files]

    all_files = pre_2017_files + post_2017_files
    all_years = pre_2017_years + post_2017_years

    relevant_files = []
    relevant_years = []
    for file, year in zip(all_files, all_years):
        filename = os.path.basename(file)
        
        relevant_files.append(file)
        relevant_years.append(year)
        
    # Make sure years are unique
    assert(len(set(relevant_years))==len(relevant_years))

    return relevant_files, relevant_years


def unify_all_inputs(data_dir, relevant_files, relevant_years):
    address_df = pd.DataFrame()
    all_filtered_df = pd.DataFrame()
    for file, year in zip(relevant_files, relevant_years):
        
        single_year_df =  pd.read_excel(file, na_filter=False)
        
        if year < 2015:
            dod_col = "DOD"
            dod_format = "%Y%m%d"
        else:
            dod_col = "DOD_4_FD"
            dod_format = "%m/%d/%Y"
            
        # add year/quarter
        if year == 2014:
            missing_date = single_year_df['death_year']=='NA'
            num_miss = np.sum(missing_date)
            print(f'{num_miss} rows in 2014 dont have a death date, {num_miss/len(single_year_df)*100:.2f}% of total')
            single_year_df = single_year_df[~missing_date]
            single_year_df['dod_dt'] = pd.to_datetime({'year':single_year_df['death_year'],
                                                    'month': single_year_df['death_month'],
                                                    'day': single_year_df['death_day']})
        else:
            missing_date = single_year_df[dod_col]==''
            num_miss = np.sum(missing_date)
            print(f'{num_miss} rows in {year} dont have a death date, {num_miss/len(single_year_df)*100:.2f}% of total')
            single_year_df = single_year_df[~missing_date]
            single_year_df['dod_dt'] = pd.to_datetime(single_year_df[dod_col], format=dod_format)
            
        single_year_df['year'] = single_year_df['dod_dt'].dt.year
        single_year_df['quarter'] = single_year_df['dod_dt'].dt.quarter

        has_ffix = year > 2014

        if has_ffix:
            address_cols = ['RES_ADDR_NUM', 'RES_STREET_PREFIX',
                        'RES_ADDR1', 'RES_STREET_DESIG',
                    'RES_STREET_SUFFIX']
            state = 'MASSACHUSETTS'
        else:
            address_cols = ['RES_ADDR_NUM',
                        'RES_ADDR1', 'RES_STREET_DESIG',
                    ]
            state = 'MA'

        if has_ffix and 'RES_STREET_PREFIX' not in single_year_df.columns:
            print(f'No decedent address column in {year}')
            #continue
            
        if year==2014:
            single_year_df[['RES_ADDR_NUM', 'RES_ADDR1', 'RES_CITY', 'RES_STATE']] = single_year_df['res_addres'].str.split(',', n=3, expand=True)
            single_year_df.loc[:,['RES_STATE']] = single_year_df['RES_STATE'].str.strip()
            single_year_df[['RES_STREET_DESIG']] = ''
            single_year_df.loc[:,['RES_ZIP']] =  single_year_df['Postal'].apply(lambda x: '0'+str(x))
            
        tot_rows = single_year_df.shape[0]


        # remove unknown address
        filtered_df = single_year_df[single_year_df['RES_ADDR1'] != 'UNKNOWN']
        filtered_df = filtered_df[filtered_df['RES_ADDR1'] != 'UNK']
        try:
            count_filtered = single_year_df['RES_ADDR1'].value_counts()['UNKNOWN']
            count_filtered += single_year_df['RES_ADDR1'].value_counts()['UNK']
        except KeyError:
            count_filtered=0
        # remove blank address
        filtered_df = filtered_df[filtered_df['RES_ADDR1'] != '']
        try:
            count_filtered += single_year_df['RES_ADDR1'].value_counts()[''] 
        except KeyError:
            count_filtered += 0

        print(f'In {year} {count_filtered} rows have missing decedent address, '
            f'{count_filtered/tot_rows*100:.1f}% of total')

        # If street number is hyphenated, take first (123-125 -> 123)
        hyphenated_rows = (filtered_df['RES_ADDR_NUM'] != filtered_df['RES_ADDR_NUM'].apply(lambda x: x.split('-')[0])).sum()
        if hyphenated_rows >0 :
            print(f'Adjusting {hyphenated_rows} hyphenated addresses in {year}.')
        filtered_df.loc[:,'RES_ADDR_NUM'] =filtered_df['RES_ADDR_NUM'].apply(lambda x: x.split('-')[0])

        # Remove letters from street number
        alphabetic_rows = (filtered_df['RES_ADDR_NUM'] != filtered_df['RES_ADDR_NUM'].str.replace(r'\D','')).sum()
        if alphabetic_rows > 0:
            print(f'Adjusting {alphabetic_rows} addresses with letters in {year}.')

        filtered_df.loc[:,'RES_ADDR_NUM'] = filtered_df['RES_ADDR_NUM'].str.replace(r'\D','')



        filtered_df.loc[:,'address'] = filtered_df[address_cols].agg(' '.join, axis=1)

        if 'SFN_NUM' not in filtered_df.columns:
            print(f'{year} is missing SFN_NUM, creating new column [year]_[row]') 
            filtered_df['SFN_NUM'] = f'{year}_' + filtered_df.index.astype(str)
            
        count_other_states = filtered_df[filtered_df['RES_STATE']!=state].shape[0]
        print(f'Ignoring {count_other_states} decedents not from  {state}')
        filtered_df = filtered_df[filtered_df['RES_STATE']==state]

        address_df = pd.concat([address_df, filtered_df[['SFN_NUM', 'address','RES_CITY', 'RES_STATE', 'RES_ZIP']]])
        all_filtered_df = pd.concat([all_filtered_df, filtered_df])

    return address_df, all_filtered_df


def geocode_addresses(address_df, output_dir):
    address_file1 = os.path.join(output_dir,'decedent_addresses_1.csv')
    address_df.iloc[:7000,:].to_csv(address_file1, index=False)
    address_file2 = os.path.join(output_dir,'decedent_addresses_2.csv')
    address_df.iloc[7000:14000,:].to_csv(address_file2, index=False)
    address_file3 = os.path.join(output_dir,'decedent_addresses_3.csv')
    address_df.iloc[14000:,:].to_csv(address_file3, index=False)

    cg = CensusGeocode(benchmark='Public_AR_Current', vintage='ACS2021_Current')

    start = time.time()

    for pt, address_file in enumerate([address_file1,
                                    address_file2,
                                    address_file3]):
        response = cg.addressbatch(address_file)
        response_df = pd.DataFrame(response)
        response_df.to_csv(os.path.join(output_dir,f'res_response_pt{pt}.csv'),index=False)
        curr = time.time()
        print(f'Elapsed: {curr-start}')

    response_1 = pd.read_csv(os.path.join(output_dir,'res_response_pt0.csv'))
    response_2 = pd.read_csv(os.path.join(output_dir,'res_response_pt1.csv'))
    response_3 = pd.read_csv(os.path.join(output_dir,'res_response_pt2.csv'))
    
    response_df = pd.concat([response_1, response_2, response_3])
    response_df.to_csv(os.path.join(output_dir,'res_response_2000_2020.csv'), index=False)

    return response_df


def match_responses(all_filtered_df, response_df):
    matched_df_no_year = response_df[response_df['match']]
    count_matched = matched_df_no_year.shape[0]
    count_filtered = response_df.shape[0]
    print(f'Matched {count_matched} rows, {count_matched/count_filtered*100:.2f}% of all filtered rows')

    matched_df_no_year.loc[:,'id'] = matched_df_no_year.loc[:,'id'].astype(str)
    all_filtered_df.loc[:,'SFN_NUM'] = all_filtered_df.loc[:,'SFN_NUM'].astype(str)
    matched_df = matched_df_no_year.merge(all_filtered_df[['SFN_NUM','year', 'quarter', 'dod_dt']], left_on='id', right_on='SFN_NUM')
    assert (len(matched_df)==len(matched_df_no_year))

    return matched_df


def make_deaths_tract_month(matched_df, mass_gdf):

    matched_df['dod_dt'] = pd.to_datetime(matched_df['dod_dt'])
    matched_df['month'] = matched_df['dod_dt'].dt.month 
        
    matched_df.loc[:,'tract'] = matched_df['tract'].astype(int)
    mass_gdf.loc[:,'TRACTCE'] = mass_gdf['TRACTCE'].astype(int)

    deaths_per_tract_df = matched_df.groupby(['year','month','tract',]).size().reset_index(name='deaths')

    no = 0
    for tract in deaths_per_tract_df.tract.unique():
        if tract not in mass_gdf.TRACTCE.unique():
            raise ValueError('Failed to match a tract!')
        
    deaths_gdf = gpd.GeoDataFrame()
    for year in deaths_per_tract_df.year.unique():
        for month in deaths_per_tract_df.month.unique():
            these_deaths = deaths_per_tract_df[(deaths_per_tract_df['year']==year) & (deaths_per_tract_df['month']==month)]
            years_merged_deaths = mass_gdf.merge(these_deaths,
                                                left_on='TRACTCE',
                                                right_on='tract',
                                                how='left')
            # fill NAs
            years_merged_deaths.loc[:,'year']=year
            years_merged_deaths.loc[:,'month']=month
            years_merged_deaths.loc[:,'deaths'] = years_merged_deaths.loc[:,'deaths'].fillna(0)
            deaths_gdf = pd.concat([deaths_gdf, years_merged_deaths])


    return deaths_gdf


def add_svi_to_data(svi_df, deaths_gdf):

    deaths_gdf.loc[:,'TRACTCE'] = deaths_gdf.loc[:,'TRACTCE'].astype(str)
    deaths_gdf.loc[:,'GEOID'] = deaths_gdf.loc[:,'GEOID'].astype(str)
    
    svi_dir = os.path.join(data_dir,'SocialVulnerabilityIndex')

    all_df = gpd.GeoDataFrame()
    for year in range(2000,2022):
        if year <= 2006:
            svi_year=2000
            theme_cols = ['MAG1TP','MAG2TP','MAG3TP','MAG4TP','MATP']
            tract_col = 'TRACT'
            geo_col = 'FIPS'
            pop_col = 'Totpop2000'
        elif year <= 2012:
            svi_year = 2010
            theme_cols = ['R_PL_THEME1','R_PL_THEME2','R_PL_THEME3','R_PL_THEME4','R_PL_THEMES']
            tract_col = 'TRACT'
            pop_col = 'E_TOTPOP'
        elif year <=2014:
            svi_year = 2014
            theme_cols = ['RPL_THEME1','RPL_THEME2','RPL_THEME3','RPL_THEME4','RPL_THEMES']
            tract_col = 'TRACTCE'
        elif year <= 2018:
            svi_year = 2016
        elif year <= 2019:
            svi_year = 2018
        else:
            svi_year = 2020
        
            
        svi_file = os.path.join(svi_dir, f'Massachusetts_SVI_{svi_year}.csv')
        svi_df = pd.read_csv(svi_file)
        
        if svi_year >= 2016:
            svi_df.loc[:,tract_col] = svi_df.FIPS.astype(str).apply(lambda x: x[5:])
        
        svi_df.loc[:,tract_col] = svi_df[tract_col].astype(str)
        svi_df.loc[:,geo_col] = svi_df[geo_col].astype(str)
        svi_df.loc[:,'ROUNDED_TRACT'] = svi_df[tract_col].astype(str).apply(lambda x: x[:-2] +'00' )
        
        target_cols = ['theme_1_pctile', 'theme_2_pctile', 'theme_3_pctile', 'theme_4_pctile', 'svi_pctile', 'pop']
        svi_df = svi_df.rename(columns={theme:target for theme, target in zip(theme_cols+[pop_col], target_cols)})

        svi_df_rounded = svi_df[target_cols+['ROUNDED_TRACT']].groupby('ROUNDED_TRACT').mean().reset_index()
        
        for month in range(1, 13):

            these_deaths = deaths_gdf[(deaths_gdf['year']==year)&(deaths_gdf['month']==month)]

            print(f'Deaths from {year} using SVI {svi_year}')

            first_merged_df = these_deaths.merge(svi_df[target_cols+[tract_col, geo_col]], left_on='GEOID', right_on=geo_col, how='left', indicator=True)
            if tract_col != 'TRACTCE':
                first_merged_df = first_merged_df.drop(columns=[tract_col])
            else:
                first_merged_df = first_merged_df.rename(columns={'TRACTCE_x':'TRACTCE'})
            unmerged_df = first_merged_df[first_merged_df['_merge']=='left_only'].drop(columns=target_cols+['_merge'])
            first_merged_df = first_merged_df[first_merged_df['_merge']=='both'].drop(columns=['_merge'])
            count_unmerged = len(unmerged_df)
            print(f"Successfully merged {(len(first_merged_df)-count_unmerged)/len(first_merged_df)*100:.1f}% on first pass")

            unmerged_df.loc[:,'ROUNDED_TRACTCE'] = unmerged_df['TRACTCE'].astype(str).apply(lambda x: x[:-2] +'00' ).astype(str)

            second_merged_df = unmerged_df.merge(svi_df_rounded[target_cols+['ROUNDED_TRACT']], left_on='ROUNDED_TRACTCE', right_on='ROUNDED_TRACT', how='left', indicator=True)
            unmerged_df = second_merged_df[second_merged_df['_merge']=='left_only'].drop(columns=target_cols+['_merge'])
            second_merged_df = second_merged_df.drop(columns=['_merge'])
            count_unmerged = len(unmerged_df)
            print(f"Successfully merged {(len(second_merged_df)-count_unmerged)/len(second_merged_df)*100:.1f}% of remaineder on second pass")

            merged_df = pd.concat([first_merged_df, second_merged_df])
            all_df = pd.concat([all_df, merged_df])

    no_data = len(all_df[all_df['svi_pctile']==-999])
    print(f'{no_data/len(all_df)*100:.1f}% of data is missing, replacing with mean')
    all_df =all_df.replace(-999, np.nan)
    all_df.loc[:,target_cols] = all_df[target_cols].fillna(all_df[target_cols].mean())

    return all_df


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    https://stackoverflow.com/a/4913653/1748679
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def quarter_cleaning(svi_gdf, output_dir):

    # Call it "grid_squar" because geopandas only supports len 10 columns
    svi_gdf = svi_gdf.rename(columns={'INTPTLAT': 'lat', 'INTPTLON': 'lon', 'GEOID': 'geoid'})
    # Make lat and lon floats
    svi_gdf.loc[:, 'lat'] = svi_gdf.lat.astype(float)
    svi_gdf.loc[:, 'lon'] = svi_gdf.lon.astype(float)
    deaths_gdf = svi_gdf

    # Used when we just need the unique tracts and their locations
    just_grid = deaths_gdf.loc[
        (deaths_gdf['year'] == 2000) & (deaths_gdf['month'] == 1), ['geoid', 'geometry', 'lat', 'lon']]

    # Calculate each squares neighbors
    neighbors = {}
    for _, row in just_grid.iterrows():
        just_grid.loc[:, 'haversine'] = just_grid.apply(lambda x: haversine(row['lon'], row['lat'],
                                                                            x['lon'], x['lat']),
                                                        axis=1)
        matching_neighbors = just_grid[just_grid['haversine'] < 8]['geoid'].values
        neighbors[row['geoid']] = matching_neighbors

    tracts = deaths_gdf['geoid'].unique()
    min_year = int(deaths_gdf.year.min())
    max_year = int(deaths_gdf.year.max())
    deaths_gdf = deaths_gdf.set_index(['geoid', 'year', 'month']).sort_index()

    month_since_2000 = 0
    season_since_2000 = 0
    qtr_since_2000 = 0
    year_since_2000 = 0
    for year in range(min_year, max_year + 1):
        for month in range(1, 12 + 1):

            if month in [1, 2, 3, 4, 5, 6]:
                season = 'jan-jun'
            else:
                season = 'jul-dec'

            if month <= 3:
                qtr = 1
            elif month <= 6:
                qtr = 2
            elif month <= 9:
                qtr = 3
            else:
                qtr = 4

            deaths_gdf.loc[idx[:, year, month], 'month_since_2000'] = month_since_2000
            deaths_gdf.loc[idx[:, year, month], 'season'] = season
            deaths_gdf.loc[idx[:, year, month], 'season_since_2000'] = season_since_2000
            deaths_gdf.loc[idx[:, year, month], 'quarter'] = qtr
            deaths_gdf.loc[idx[:, year, month], 'qtr_since_2000'] = qtr_since_2000
            deaths_gdf.loc[idx[:, year, month], 'year_since_2000'] = year_since_2000

            month_since_2000 += 1

            if month in [6, 12]:
                season_since_2000 += 1

            if month in [3, 6, 9, 12]:
                qtr_since_2000 += 1

            if month == 12:
                year_since_2000 += 1

    deaths_gdf = deaths_gdf.reset_index()

    cleaned_gdf = deaths_gdf.set_index(['geoid', 'year', 'quarter']).sort_index()
    cleaned_gdf.loc[idx[:, :, :], 'self_t-1'] = cleaned_gdf.loc[idx[:, :, :], 'deaths'].shift(1, fill_value=0)
    unduped_gdf = cleaned_gdf[~cleaned_gdf.index.duplicated(keep='first')]
    summed_deaths = cleaned_gdf[['deaths']].groupby(level=[0,1,2]).sum()[['deaths']]
    summed_deaths = summed_deaths.merge(unduped_gdf, how='left', left_index=True, right_index=True,suffixes=[None,'_garbage'])
    summed_deaths = summed_deaths.drop('deaths_garbage',axis=1)
    cleaned_gdf = summed_deaths
    for tract in tracts:
        cleaned_gdf.loc[idx[tract, :, :], 'neighbor_t-1'] = \
            cleaned_gdf.loc[idx[neighbors[tract], :, :], 'self_t-1'].groupby(level=['year', 'quarter']).mean().shift(1,
                                                                                                                    fill_value=0).values

    timestep = 0

    for year in range(min_year, max_year + 1):
        for quarter in range(1, 5):
            cleaned_gdf.loc[idx[:, year, quarter], 'timestep'] = timestep
            timestep += 1

    cleaned_gdf = cleaned_gdf.reset_index()

    svi_out_file = os.path.join(output_dir, 'clean_quarter_tract')
    gpd.GeoDataFrame(cleaned_gdf).to_file(svi_out_file)

    cleaned_gdf = deaths_gdf.set_index(['geoid', 'year', 'season']).sort_index()
    cleaned_gdf.loc[idx[:, :, :], 'self_t-1'] = cleaned_gdf.loc[idx[:, :, :], 'deaths'].shift(1, fill_value=0)
    unduped_gdf = cleaned_gdf[~cleaned_gdf.index.duplicated(keep='first')]
    summed_deaths = cleaned_gdf[['deaths']].groupby(level=[0,1,2]).sum()[['deaths']]
    summed_deaths = summed_deaths.merge(unduped_gdf, how='left', left_index=True, right_index=True,suffixes=[None,'_garbage'])
    summed_deaths = summed_deaths.drop('deaths_garbage',axis=1)
    cleaned_gdf = summed_deaths
    for tract in tracts:
        cleaned_gdf.loc[idx[tract, :, :], 'neighbor_t-1'] = \
            cleaned_gdf.loc[idx[neighbors[tract], :, :], 'self_t-1'].groupby(level=['year', 'season']).mean().shift(1,
                                                                                                                    fill_value=0).values

    timestep = 0

    for year in range(min_year, max_year + 1):
        for season in ['jan-jun', 'jul-dec']:
            cleaned_gdf.loc[idx[:, year, season], 'timestep'] = timestep
            timestep += 1

    cleaned_gdf = cleaned_gdf.reset_index()

    svi_out_file = os.path.join(output_dir, 'clean_semi_tract')
    gpd.GeoDataFrame(cleaned_gdf).to_file(svi_out_file)

    
    cleaned_gdf = deaths_gdf.set_index(['geoid', 'year']).sort_index()
    cleaned_gdf.loc[idx[:, :, :], 'self_t-1'] = cleaned_gdf.loc[idx[:, :], 'deaths'].shift(1, fill_value=0)
    unduped_gdf = cleaned_gdf[~cleaned_gdf.index.duplicated(keep='first')]
    summed_deaths = cleaned_gdf[['deaths']].groupby(level=[0,1]).sum()[['deaths']]
    summed_deaths = summed_deaths.merge(unduped_gdf, how='left', left_index=True, right_index=True,suffixes=[None,'_garbage'])
    summed_deaths = summed_deaths.drop('deaths_garbage',axis=1)
    cleaned_gdf = summed_deaths
    for tract in tracts:
        cleaned_gdf.loc[idx[tract, :], 'neighbor_t-1'] = \
            cleaned_gdf.loc[idx[neighbors[tract], :], 'self_t-1'].groupby(level=['year']).mean().shift(1, fill_value=0).values

    timestep = 0

    for year in range(min_year, max_year + 1):
            cleaned_gdf.loc[idx[:, year], 'timestep'] = timestep
            timestep += 1

    cleaned_gdf = cleaned_gdf.reset_index()

    svi_out_file = os.path.join(output_dir, 'clean_annual_tract')
    gpd.GeoDataFrame(cleaned_gdf).to_file(svi_out_file)

    return cleaned_gdf



def process_data(data_dir, output_dir, do_geocoding, make_monthly, add_svi, do_final_quarterly):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mass_shapefile = os.path.join(data_dir,'shapefiles','MA_2021')
    mass_gdf = gpd.read_file(mass_shapefile)

    if do_geocoding:
        relevant_files, relevant_years = get_available_files_and_years(data_dir)

        address_df, all_filtered_df = unify_all_inputs(data_dir, relevant_files, relevant_years)

        response_df = geocode_addresses(address_df, output_dir)

        matched_df = match_responses(all_filtered_df, response_df)

        matched_df.to_csv(os.path.join(output_dir,'geocoded_deaths_2000_2021.csv'), index=False)

    deaths_file = os.path.join(output_dir,'res_deaths_month_all')
    if make_monthly:
        if not do_geocoding:
            matched_df = pd.read_csv(os.path.join(output_dir,'geocoded_deaths_2000_2021.csv'))
        deaths_gdf = make_deaths_tract_month(matched_df, mass_gdf)
        deaths_gdf.to_file(deaths_file)


    svi_file = os.path.join(output_dir,'svi_month')
    if add_svi:
        if not make_monthly:
            deaths_gdf = gpd.read_file(deaths_file)
        all_df = add_svi_to_data(data_dir, deaths_gdf)
        gpd.GeoDataFrame(all_df).to_file(svi_file)


    if do_final_quarterly:
        if not add_svi:
            all_df = gpd.read_file(svi_file)
        cleaned_gdf = quarter_cleaning(all_df, output_dir)
        svi_out_file = os.path.join(output_dir, 'clean_quarter_tract')
        gpd.GeoDataFrame(cleaned_gdf).to_file(svi_out_file)

    return




if __name__ == '__main__':

    parser  = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help="Directory to save output in")
    parser.add_argument('--do_geocoding', action='store_true', help="Whether to do geocoding, if not, will load data from output_dir")
    parser.add_argument('--make_monthly', action='store_true', help="Whether to do the step to make deaths monthly, if not present will skip")
    parser.add_argument('--add_svi', action='store_true', help="Whether to do the step to make deaths monthly, if not present will skip")
    parser.add_argument('--do_final_quarterly', action='store_true', help="Process to quarters")
    args = parser.parse_args()

    data_dir = os.environ.get('DATA_DIR', None)
    if data_dir is None or not os.path.exists(data_dir):
        raise ValueError("Please set DATA_DIR")
    

    process_data(data_dir, args.output_dir, args.do_geocoding, args.make_monthly, args.add_svi, args.do_final_quarterly)
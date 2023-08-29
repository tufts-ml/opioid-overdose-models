import os
import argparse
from glob import glob
import time

import numpy as np
import pandas as pd
import censusgeocode as cg
import geopandas as gpd


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
            print(f'No decdent address column in {year}')
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

        print(f'In {year} {count_filtered} rows have missing decdent address, '
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
        print(f'Ignoring {count_other_states} decdencts not from  {state}')
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



def process_data(data_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    relevant_files, relevant_years = get_available_files_and_years(data_dir)

    address_df, all_filtered_df = unify_all_inputs(data_dir, relevant_files, relevant_years)

    response_df = geocode_addresses(address_df, output_dir)

    matched_df = match_responses(all_filtered_df, response_df)

    matched_df.to_csv(os.path.join(output_dir,'geocoded_deaths_2000_2020.csv'), index=False)


if __name__ == '__main__':

    parser  = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help="Directory to save output in")
    args = parser.parse_args()

    data_dir = os.environ.get('DATA_DIR', None)
    if data_dir is None or not os.path.exists(data_dir):
        raise ValueError("Please set DATA_DIR")
    

    process_data(data_dir, args.output_dir)

This folder pre-processes data from the Medical Examiner's website for Cook County, IL. 
Raw data can be found in this directory and at the following links

ME_CSV_and_Shapefile: https://maps.cookcountyil.gov/medexammaps/
SVI: https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html 
TigerLine Shapefiles: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html 


To recreate the dataset, run `extract_dataset.py`

To recreate the dataset from a single .py file, run `cook_county_total_script.py`. The file also comes with a custom_timestep_range optional command line argument, where --custom_timestep_range "n" can be added to the command. Here, n is an integer, and the script will create a dataset that has timestep year / n. 
Example: `cook_county_total_script.py --custom_timestep_range 12` will create a monthly dataframe. 
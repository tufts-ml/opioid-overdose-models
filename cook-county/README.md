@author - Jyontika Kapoor (jk103@wellesley.edu)
Under advisement of Kyle Heuton and Michael C. Hughes

PrjecT: pioid-Model Forecasting 

This folder pre-processes data from the Medical Examiner's website for Cook County, IL. 
Raw data can be found on hughes-lab cluster, and links are below as well:

ME_CSV_and_Shapefile: https://maps.cookcountyil.gov/medexammaps/
SVI: https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html 
TigerLine Shapefiles: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html 

---
Note that all notebooks write to same CSV files: (i) annual, (ii) quarterly, (iii) semiannually
The order in which I run these notebooks is: intro --> annual, quarter, semi --> SVI

--
cook-county-experiment-runner calculates BPR, RMSE, and MAE for cook county

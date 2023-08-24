import pandas as pd
import numpy as np
import os

def int2yes(i):
    return 'yes' if i > 0 else 'no'


def verify(data_dir):

    comparison_list = [
        ("Year and Semi->Year", 
             "cook_county_gdf_cleanwithsvi_year.csv",
             "cook_county_gdf_cleanwithsvi_semiannual.csv",
             ['geoid', 'year'], ['geoid', 'year', 'deaths']),
        ("Year and Quarter->Year", 
             "cook_county_gdf_cleanwithsvi_year.csv",
             "cook_county_gdf_cleanwithsvi_quarterly.csv",
             ['geoid', 'year'], ['geoid', 'year', 'deaths']),
        ("Semi and Quarter->Semi", 
             "cook_county_gdf_cleanwithsvi_semiannual.csv",
             "cook_county_gdf_cleanwithsvi_quarterly.csv",
             ['geoid', 'year', 'semiannual'], ['geoid', 'year', 'semiannual', 'deaths']),
        ]

    for label, csvA, csvB, gby_cols, match_cols in comparison_list:
    
        big_df = pd.read_csv(os.path.join(data_dir, csvA))
        mini_df = pd.read_csv(os.path.join(data_dir, csvB))
        assert big_df.shape[0] < mini_df.shape[0]
        
        big2_df = mini_df.groupby(gby_cols).agg({'deaths': 'sum'}).reset_index()

        print("\nVerifying %s" % label)
        print("df A: shape %10s  csv %s" % (str(big_df.shape), csvA))
        print("df B: shape %10s  csv %s" % (str(mini_df.shape), csvB))

        valA = big_df['deaths'].sum()
        valB = mini_df['deaths'].sum()
        print("  %s : do total deaths agree? A=%05d B=%05d" % (int2yes(valA == valB), valA, valB))

        matches = np.all(big_df[match_cols].values == big2_df[match_cols].values, axis=0)
        for cc, col in enumerate(match_cols):
            msg = "  %s : do all values in col %s match?" % (int2yes(matches[cc]), col)
            print(msg)
        print(big_df[match_cols].head())
        print(big2_df[match_cols].head())
        assert np.all(matches)
        print(big_df['deaths'].sum())

if __name__ == '__main__':
    data_dir = os.environ.get('DATA_DIR', '/Users/jyontika/Desktop/cook-county/data/')

    verify(data_dir)
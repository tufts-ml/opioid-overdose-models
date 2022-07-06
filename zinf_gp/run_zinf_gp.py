"""Run a zero-inflated GP on opioid data"""
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
idx = pd.IndexSlice
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import copy

import geopandas as gpd

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

import gpflow
import tensorflow as tf

from math import radians, cos, sin, asin, sqrt

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


def run_model(time=None, data_dir=None):

    result_dir = os.path.join(data_dir, 'results')
    mass_shapefile = os.path.join(data_dir, 'shapefiles', 'MA_2021')

    svi_file = os.path.join(result_dir, 'svi_month')
    svi_gdf = gpd.read_file(svi_file)
    # Call it "grid_squar" because geopandas only supports len 10 columns
    svi_gdf = svi_gdf.rename(columns={'INTPTLAT': 'lat', 'INTPTLON': 'lon', 'GEOID': 'grid_squar'})
    # Make lat and lon floats
    svi_gdf.loc[:, 'lat'] = svi_gdf.lat.astype(float)
    svi_gdf.loc[:, 'lon'] = svi_gdf.lon.astype(float)
    deaths_gdf = svi_gdf

    # Used when we just need the unique tracts and their locations
    just_grid = deaths_gdf.loc[
        (deaths_gdf['year'] == 2000) & (deaths_gdf['month'] == 1), ['grid_squar', 'geometry', 'lat', 'lon']]

    # Calculate each squares neighbors
    neighbors = {}
    for _, row in just_grid.iterrows():
        just_grid.loc[:, 'haversine'] = just_grid.apply(lambda x: haversine(row['lon'], row['lat'],
                                                                            x['lon'], x['lat']),
                                                        axis=1)
        matching_neighbors = just_grid[just_grid['haversine'] < 8]['grid_squar'].values
        neighbors[row['grid_squar']] = matching_neighbors

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser

    parser.add_argument('--time', type=str, help="Temporal division of data",
                        choices=['qtr','biannual', 'annual'])
    parser.add_argument('--data_dir', type=str, help="Path to opioid data",
                        default='/cluster/tufts/hugheslab/datasets/NSF_OD/')
    
""" Functions for cleaning tasks: merging, transforming, averaging nighlights data."""

import pandas as pd
import numpy as np
import geoio
from cnn_satellite.helper_codes import aux_functions


def data_prep(data_cons,data_geo):
    """Produce relevant data set - transform consumption and convert household
       level data to household-cluster level data.

    Args:
        data (pandas.DataFrame): The two .dta format data sets.

    Returns
    -------
        pandas.DataFrame: The cleaned data set.

    """
    
    # Transform consumption, adjusted by PPP
    data_cons['cons'] = data_cons['rexpagg']/(365*data_cons['adulteq'])/215.182  
    df_cons = data_cons[['case_id','hh_wgt','hhsize','cons']]     # only need certain variables
    df_coords = data_geo[['case_id','lat_modified','lon_modified']]
 
    df_merged = pd.merge(df_cons, df_coords, on='case_id')   # merge variables from different data sets by case identifier 
    df_merged.dropna(inplace=True) 
    
    df_clusters = df_merged.groupby(['lon_modified', 'lat_modified']).sum(numeric_only=True).reset_index()
    df_clusters['cons_pc'] = df_clusters['cons'] / df_clusters['hhsize'] # cons per capita - divides cluster income by HH size
    df_processed = df_clusters[['lon_modified', 'lat_modified', 'cons_pc']] 
    
    return df_processed


def nightlights_prep(df):
    
    """ For every coordinate in dataframe, find the mean nighlight in an s x s 
        km  square grid around that coordinate. 

    Args:
        data (pandas.DataFrame): Processed survey data.

    Returns
    -------
        pandas.DataFrame: Processed survey data with nightlight means .

    """
    tif = [geoio.GeoImage('./src/cnn_satellite/data/nightlights_data/viirs_2015.tif')][0]
    tif_array = np.squeeze(tif.get_data())

    cluster_nightlights = []
    for i,r in df.iterrows():
        min_lat, min_lon, max_lat, max_lon = aux_functions.grid_generator(r.lat_modified, r.lon_modified) 
        xminPixel, ymaxPixel = tif.proj_to_raster(min_lon, min_lat)
        xmaxPixel, yminPixel = tif.proj_to_raster(max_lon, max_lat)
        xminPixel, yminPixel, xmaxPixel, ymaxPixel = int(xminPixel), int(yminPixel), int(xmaxPixel), int(ymaxPixel)
        cluster_nightlights.append(tif_array[yminPixel:ymaxPixel,xminPixel:xmaxPixel].mean())
        
    df['nightlights'] = cluster_nightlights
    malawi_data = df
    
    return malawi_data



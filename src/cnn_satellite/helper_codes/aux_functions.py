import math
import random
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
from sklearn.decomposition import PCA

def grid_generator(lat, lon, s=10):
    
    """ Generates a 10 km x 10 km square grid with center on (latitude, longitude).

    Args:
        lat (float): Coordinate latitude  
        lon (float): Coordinate longitude
        s (float): Grid size

    Returns
    -------
        list: four coordinates of grid.

    """
   
    v = (180/math.pi)*(500/6378137)*s 
    grid_coord = (lat - v, lon - v, lat + v, lon + v)
    
    return grid_coord

def data_ML_feed(df):
     
    # Load data
    Y = df['cons_pc'].values
    Y = np.log(Y).reshape(-1,1)

    X_nlight = df['nightlights'].values.reshape(780,1)
    X = np.array(df.drop(['cons_pc','nightlights'], axis = 1))
    X = MinMaxScaler().fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca)
    X_pca = X_pca.loc[:, 0:9]
    
    return Y, X_nlight, X_pca 
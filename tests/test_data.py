import numpy as np
import pandas as pd
import pytest
import os

from cnn_satellite.config import BLD, SRC
from cnn_satellite.data_management import clean_data


def test_downloaded_images():
    
    # Check if downloaded and unzipped sat data retained all images.
    lst = os.listdir("./src/cnn_satellite/data/sat_images/")
    num_cols = pd.read_stata("./bld/python/data/image_locations_bin.dta")
    assert len(lst) == num_cols.shape[0], f"Downloaded images are {len(lst)} but locations are {num_cols.shape[0]}."
    

def test_malawi_data_dropna():
    
    # does processed and merged malawi data contain any NAs
    data_cons = pd.read_stata("./src/cnn_satellite/data/survey_data/ihs4 consumption aggregate.dta")
    data_geo = pd.read_stata("./src/cnn_satellite/data/survey_data/householdgeovariablesihs4.dta")
    
    data_clean = clean_data.data_prep(data_cons,data_geo)
    assert not data_clean.isna().any(axis=None), "Unfortunately, there are NAs in the Malawi dataset."
    
def test_GMM_bins():
    
    # Does symlink copy the correct number of images to the GMM-based nightlight bins?
    classes = ['NO_LIGHT', 'LOW_LIGHT', 'HIGH_LIGHT']
    image_locs_bins = pd.read_stata("./bld/python/data/image_locations_bin.dta")
    im_exp_train = len(image_locs_bins.loc[image_locs_bins['train'] == 1])
    im_exp_val = len(image_locs_bins.loc[image_locs_bins['train'] == 0]) 

    im_train = 0
    im_val = 0
    for c in classes:
        im_train += len(os.listdir(os.path.join("./bld/python/data/train", str(c))))
        im_val += len(os.listdir(os.path.join("./bld/python/data/val", str(c))))

    assert im_train==im_exp_train, f"There are {im_train} images in the train folder, expected {im_exp_train}."
    assert im_val==im_exp_val, f"There are {im_val} images in the train folder, expected {im_exp_val}."


"""Tasks for downloading, extracting, and copying the data. 
   Uses os.symlink to save space when copying.
   
"""

import pandas as pd
import pytask
import os
import zipfile
import gdown

from cnn_satellite.config import BLD, SRC
from cnn_satellite.data_management import clean_data
from cnn_satellite.utilities import read_yaml

@pytask.mark.persist
@pytask.mark.produces(
    {
    "sat_images_zip": SRC / "data" / "sat_images.zip",
    "viirs_2015": SRC / "data" / "nightlights_data" / "viirs_2015.tif",
    }
)

def task_download_data(produces):
    
    # Download satellite imagery and nightlights data from Google Drive and store it to designated folders.
    url_sat = 'https://drive.google.com/u/0/uc?id=1wEV4w3GTB2UL4RUUrSJSm4ImOzq40A_G'
    gdown.download(url_sat, './src/cnn_satellite/data/sat_images.zip', quiet=True)
    
    url_night = 'https://drive.google.com/u/0/uc?id=1X65wI7gH0ISTDAkVpjfUGvsQuX5g23mH'
    gdown.download(url_night, './src/cnn_satellite/data/nightlights_data/viirs_2015.tif', quiet=True)
    
    pass

@pytask.mark.depends_on(
    {
    "sat_images_zip": SRC / "data" / "sat_images.zip",
    }
)

@pytask.mark.produces(
    {
    "sat_images": SRC / "data",
    }
)

def task_extract_data(depends_on,produces):
   
    # Unzip the compressed satellite imagery data.
    with zipfile.ZipFile(depends_on["sat_images_zip"], 'r') as zip_ref:
        zip_ref.extractall(produces["sat_images"])
    
    pass

@pytask.mark.depends_on(
    {
        "cons": SRC / "data" / "survey_data" / "ihs4 consumption aggregate.dta",
        "geo": SRC / "data" / "survey_data" / "householdgeovariablesihs4.dta",
        "viirs_2015": SRC / "data" / "nightlights_data" /"viirs_2015.tif",
    }
)
@pytask.mark.produces(
    {
    "nlight_proc": BLD / "python" / "data" / "malawi.dta",
    }
)

def task_clean_data_python(depends_on, produces):
    
    # Call the data cleaner and create new dataset using nightlights data.
    data_cons = pd.read_stata(depends_on["cons"])
    data_geo = pd.read_stata(depends_on["geo"])

    data_processed = clean_data.data_prep(data_cons,data_geo) # convert from household-level to HH-cluster level data
    malawi_data = clean_data.nightlights_prep(data_processed)  # add mean nightlights of HH clusters
    malawi_data.to_stata(produces["nlight_proc"], write_index = False)
    pass
    

@pytask.mark.persist 
@pytask.mark.depends_on(
    {
        "image_locs_bin": BLD / "python" / "data" / "image_locations_bin.dta",
        "sat_images": SRC / "data",
    }
)
@pytask.mark.produces(
    {
        "train_set": BLD / "python" / "data" / "train",
        "val_set": BLD / "python" / "data" / "val",
    } 
)
def task_create_sat_bins(depends_on,produces):
    
    """ Separate satellite images into bins depending on the different 
        nightlight classes: No light, low light, high light.  

    Args:
        data (pandas.DataFrame): Survey data with image locations and their corresponding light classes

    """
    
    nlights_class = ['NO_LIGHT', 'LOW_LIGHT', 'HIGH_LIGHT']
    
    image_locs_bins = pd.read_stata(depends_on["image_locs_bin"])
    
    train_set = image_locs_bins.loc[image_locs_bins['train'] == 1]
    val_set   = image_locs_bins.loc[image_locs_bins['train'] == 0]
    
    for i in nlights_class:
        os.makedirs(os.path.join(produces["train_set"], i),exist_ok=True)
        os.makedirs(os.path.join(produces["val_set"], i),exist_ok=True)
    
    for image, nlight_bin in zip(train_set['image_name'], train_set['nightlights_bin']):
        source_folder = os.path.abspath(os.path.join(depends_on["sat_images"],'sat_images',image))
        destination_folder = os.path.join(produces["train_set"], str(nlight_bin), image)
        os.symlink(source_folder, destination_folder, target_is_directory = False)

    for image, nlight_bin in zip(val_set['image_name'], val_set['nightlights_bin']):
        source_folder = os.path.abspath(os.path.join(depends_on["sat_images"],'sat_images',image))
        destination_folder = os.path.join(produces["val_set"], str(nlight_bin), image)
        os.symlink(source_folder, destination_folder, target_is_directory = False)
        
    pass
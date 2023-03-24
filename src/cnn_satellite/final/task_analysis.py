"""Tasks for training ML models, GMM model, and CNN model."""

import pandas as pd
import numpy as np
import pytask
import torch
import pickle

from cnn_satellite.config import BLD, SRC
from cnn_satellite.analysis import ML_pred, cnn_training, cnn_extract_features
from cnn_satellite.helper_codes import aux_functions
from cnn_satellite.utilities import read_yaml


@pytask.mark.depends_on(
    {
        "image_locs": SRC / "data" / "survey_data" / "image_locations.dta",
    }
)
@pytask.mark.produces(
    {
        "image_locs_bin": BLD / "python" / "data" / "image_locations_bin.dta",
    } 
)

def task_GMM(depends_on, produces):
    
    # Train a GMM model to create nightlight bins and assign each image file to a bin in a new dataset
    data = pd.read_stata(depends_on["image_locs"])
    dataset_bin = ML_pred.GMM_estim(data)
    dataset_bin.to_stata(produces["image_locs_bin"], write_index = False)   # create dataset with bins
    
    pass


@pytask.mark.persist
@pytask.mark.depends_on(
    {
        "train_set": BLD / "python" / "data" / "train",
        "val_set": BLD / "python" / "data" / "val",
    } 
)

@pytask.mark.produces(
    {
        "cnn_model": BLD / "python" / "model" / "cnn_model.pt",
        "cnn_hist": BLD / "python" / "model" / "cnn_hist.csv",
    }
)
def task_CNN_train(depends_on,produces):
    
    # Execute CNN model transfer learning
    cnn_model, cnn_history = cnn_training.model_execute(epochs=5)
    cnn_hist = pd.DataFrame({"Validation Accuracy": np.array(cnn_history)})

    cnn_hist.to_csv(produces["cnn_hist"]) 
    torch.save(cnn_model, produces["cnn_model"])
    pass

@pytask.mark.depends_on(
    {
        "VAL_DIR": BLD / "python" / "data" / "val",
        "image_locs_bin": BLD / "python" / "data" / "image_locations_bin.dta",
        "cnn_model": BLD / "python" / "model" / "cnn_model.pt"
    }
)

@pytask.mark.persist
@pytask.mark.produces(
    {
        "features_data": BLD / "python" / "data" / "features_data.dta",
    } 
)
    
def task_feature_extraction(depends_on, produces): 
    
    # Hook outputs from the final layer of the CNN when doing a forward pass on validation images.
    image_locs_bin = pd.read_stata(depends_on["image_locs_bin"])
    features_data = cnn_extract_features.feature_extract(depends_on["VAL_DIR"],
                                                         depends_on["cnn_model"],
                                                         image_locs_bin)
    features_data.to_stata(produces["features_data"])
    pass
    
@pytask.mark.depends_on({"features_data": BLD / "python" / "data" / "features_data.dta",})
@pytask.mark.produces(
    {
        "ML_estims": BLD / "python" / "tables" / "table_ML_estims.tex",
        "CNN_estims": BLD / "python" / "tables" / "table_CNN_estims.tex",
    } 
)
    
def task_ML_prediction(depends_on, produces): 
    
    # Train various models to predict consumption from nighlight data and save the R2 estimate in a TeX file
    data = pd.read_stata(depends_on["features_data"])
    Y, X_nlight, X_pca = aux_functions.data_ML_feed(data)
    
    estim_data_ML, estim_data_CNN = ML_pred.ML_estim_func(Y, X_pca, X_nlight)
    estim_data_ML.to_latex(buf = produces["ML_estims"]) 
    estim_data_CNN.to_latex(buf = produces["CNN_estims"])
 
    pass
    

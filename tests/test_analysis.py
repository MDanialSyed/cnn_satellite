import numpy as np
import pandas as pd
import pytest
import os

from cnn_satellite.config import BLD, SRC
from cnn_satellite.data_management import clean_data
from cnn_satellite.analysis import ML_pred
from cnn_satellite.helper_codes import aux_functions

def test_ML_nlight():
    
    # Check if predicted consumption values are replicated 
    data = pd.read_stata("./bld/python/data/features_data.dta")
    Y, X_nlight, X_pca = aux_functions.data_ML_feed(data)
    estim_data_ML, estim_data_CNN = ML_pred.ML_estim_func(Y, X_pca, X_nlight)

    assert (estim_data_ML["5-Fold"] == [0.24, 0.13, 0.24, 0.24, 0.09]).all(),"Unanticipated predicted values detected."   

        
def test_CNN_acc():
    
    # Check if you replicated the same CNN validation accuracy values.
    
    data = pd.read_csv("./bld/python/model/cnn_hist.csv")
    data = np.round(data,2)
    assert (data["Validation Accuracy"] == [0.70, 0.71, 0.70, 0.74, 0.75]).all(),"Unanticipated validation accuracies detected."    
    
    
def test_ML_CNN():
    
    # Check if predicted consumption values are replicated 
    data = pd.read_stata("./bld/python/data/features_data.dta")
    Y, X_nlight, X_pca = aux_functions.data_ML_feed(data)
    estim_data_ML, estim_data_CNN = ML_pred.ML_estim_func(Y, X_pca, X_nlight)

    assert (estim_data_ML["5-Fold"] == [0.24, 0.13, 0.24, 0.24, 0.09]).all(),"Unanticipated predicted values detected."    
    assert (estim_data_CNN["5-Fold"] == [0.21, 0.15, 0.23, 0.21, -0.10]).all(),"Unanticipated predicted values detected."  
 
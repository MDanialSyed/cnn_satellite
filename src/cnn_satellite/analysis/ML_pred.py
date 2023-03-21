""" Creates GMM nightlight bins and generates R-Square estimates for various ML models 
    to predict consumption using nightlight data
    
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from cnn_satellite.helper_codes import aux_functions
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import cross_validate


def GMM_estim(df):

    X = df['nightlights'].values.reshape(-1,1)
    gmm_fit = GMM(n_components=3).fit(X)    # fit GMM 
    labels = gmm_fit.predict(df['nightlights'].values.reshape(-1,1)) # obtain predicted labels

    thresholds = sorted([df['nightlights'][labels==0].max(),      # use maxima as nightlight bin thresholds
                         df['nightlights'][labels==1].max(), 
                         df['nightlights'][labels==2].max()], reverse=True)
    
    labels = ['HIGH_LIGHT','LOW_LIGHT','NO_LIGHT']
    df['nightlights_bin'] = 'HIGH_LIGHT'
    for t, l in zip(thresholds, labels):
        df['nightlights_bin'].loc[df['nightlights'] <= t] = l
        
    return df


def ML_estim_func(Y, X_pca, X_nlight):

    np.random.seed(27) # reproducibility 
    
    # Specify models to be used
    models = [Ridge(), Lasso(),
              RandomForestRegressor(random_state=77), 
              BayesianRidge(),
              LinearRegression(),
              XGBRegressor(learning_rate=0.015, random_state=77)]

    estimates_5k_1 = np.zeros(6)
    estimates_10k_1 = np.zeros(6)
    estimates_max_1 = np.zeros(6)

    estimates_5k_2 = np.zeros(6)
    estimates_10k_2 = np.zeros(6)
    estimates_max_2 = np.zeros(6)

    # for loop to run CV on each ML model
    for i in range(6):
        cv_results_1 = cross_validate(models[i], X_nlight, Y.ravel(), cv=5, scoring=('r2'),n_jobs=4)
        cv_results_2 = cross_validate(models[i], X_nlight, Y.ravel(), cv=10, scoring=('r2'),n_jobs=4)

        estimates_5k_1[i]  = round(np.mean(cv_results_1['test_score']),2)
        estimates_10k_1[i] = round(np.mean(cv_results_2['test_score']),2)
        estimates_max_1[i] = round(np.array((cv_results_1['test_score'].max(), cv_results_2['test_score'].max())).max(),2)

        cv_results_1 = cross_validate(models[i], X_pca, Y.ravel(), cv=5,scoring=('r2'),n_jobs=4)
        cv_results_2 = cross_validate(models[i], X_pca, Y.ravel(), cv=10,scoring=('r2'),n_jobs=4)

        estimates_5k_2[i]  = round(np.mean(cv_results_1['test_score']),2)
        estimates_10k_2[i] = round(np.mean(cv_results_2['test_score']),2)
        estimates_max_2[i] = round(np.array((cv_results_1['test_score'].max(), cv_results_2['test_score'].max())).max(),2)

        
    dataset_ML = pd.DataFrame({'5-Fold': estimates_5k_1, 
                            '10-Fold': estimates_10k_1,
                            'Maximum': estimates_max_1}, columns=['5-Fold', '10-Fold','Maximum'],
                             index = ['Ridge', 'LASSO', 'Random Forest', 'Bayesian Ridge', 'Linear Regression', 'XGBoost'])


    dataset_CNN = pd.DataFrame({'5-Fold': estimates_5k_2, 
                            '10-Fold': estimates_10k_2,
                            'Maximum': estimates_max_2}, columns=['5-Fold', '10-Fold','Maximum'],
                             index = ['Ridge', 'LASSO', 'Random Forest', 'Bayesian Ridge', 'Linear Regression', 'XGBoost'])
  
    return dataset_ML, dataset_CNN


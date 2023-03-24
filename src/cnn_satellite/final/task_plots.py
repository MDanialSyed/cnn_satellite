"""Tasks for creating plots from data."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pytask
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import  MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.linear_model import Ridge, RidgeCV
from cnn_satellite.config import BLD, SRC
from cnn_satellite.helper_codes import aux_functions
from cnn_satellite.utilities import read_yaml

@pytask.mark.depends_on({"nlight_proc": BLD / "python" / "data" / "malawi.dta"})
@pytask.mark.produces({"cons_nlight": BLD / "python" / "figures" / "cons_nlight.png"})

def task_cons_nlight(depends_on, produces):

    # Relationship between nightlights and consumption - create scatterplot with regression line
    df = pd.read_stata(depends_on["nlight_proc"])
    plt.figure(figsize=(8,5))
    sns.regplot(x='nightlights', y='cons_pc', data=df, ci=None)
    plt.ylabel('Consumption Per Capita',fontsize = 12);plt.xlabel('Mean Nightlight Intensity',fontsize = 12)
    plt.ylim(0, 4)
    plt.xlim(0, 10)
    plt.title('Average Nightlight Intensity and Consumption',fontsize = 15)
    plt.savefig(produces["cons_nlight"], bbox_inches='tight')
    plt.clf()
    pass


@pytask.mark.depends_on({"image_locs_bin": BLD / "python" / "data" / "image_locations_bin.dta"})
@pytask.mark.produces({"nlight_density": BLD / "python" / "figures" / "nlight_density.png"})

def task_GMM_nlight(depends_on, produces):
    
    #Create density plot for each nightlight bin 
    df = pd.read_stata(depends_on["image_locs_bin"])
    
    df_high = df.loc[df['nightlights_bin'] == 'HIGH_LIGHT']
    df_medium = df.loc[df['nightlights_bin'] == 'LOW_LIGHT']
    df_low = df.loc[df['nightlights_bin'] == 'NO_LIGHT']

    plt.figure(figsize=(8,6))
    plt.rcParams.update({'font.size': 16})
    sns.kdeplot(df_low.nightlights, label='low')
    sns.kdeplot(df_medium.nightlights, label='medium')
    sns.kdeplot(df_high.nightlights,label='high')

    plt.legend()
    plt.ylim(0, 1.0)
    plt.rcParams['axes.grid'] = True
    plt.xlabel('Nightlight Intensity',fontsize = 12)
    plt.ylabel('Density',fontsize = 12)
    plt.title('Density of Nightlight Intensity by Class',fontsize = 15)
    plt.savefig(produces["nlight_density"], bbox_inches='tight')
    plt.clf()
    pass

@pytask.mark.depends_on({"features_data": BLD / "python" / "data" / "features_data.dta",})
@pytask.mark.produces({"pca_fit": BLD / "python" / "figures" / "pca.png"})
def task_feature_pca(depends_on, produces):
    
    df = pd.read_stata(depends_on["features_data"])
    X = np.array(df.drop(['cons_pc','nightlights'], axis = 1))
    X = MinMaxScaler().fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X)

    # Cumulative sum of eigenvalues; This will be used to create step plot
    # for visualizing the variance explained by each principal component.
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    
    # Create the visualization plot
    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio',fontsize = 12)
    plt.xlabel('Principal component index',fontsize = 12)
    plt.xlim(0, 100)
    plt.legend(loc='best',prop={'size': 12})
    plt.savefig(produces["pca_fit"], bbox_inches='tight')
    plt.clf()
    pass

@pytask.mark.depends_on({"features_data": BLD / "python" / "data" / "features_data.dta",})
@pytask.mark.produces({"cnn_fit": BLD / "python" / "figures" / "cnn_fit.png"})

def task_cnn_ridge(depends_on, produces):
    
    #Predicting consumption with CNN features and Ridge - Create scatter plot with R2
    data = pd.read_stata(depends_on["features_data"])
    Y, _, X_pca = aux_functions.data_ML_feed(data)
   
    np.random.seed(27)
    yhat = cross_val_predict(Ridge(),X_pca,Y.ravel(),cv=5)
    rsq = round(r2_score(Y.ravel(), yhat), 2)

    plt.figure(figsize=(8,5))
    sns.regplot(x=np.exp(Y), y=np.exp(yhat),ci=None)
    plt.ylabel('Predicted Consumption',fontsize = 12);plt.xlabel('Actual Consumption',fontsize = 12)
    plt.ylim(0, 3)
    plt.xlim(0, 5)
    plt.text(4, 1, f'r^2={round(rsq, 2)}', size=12)
    plt.title('Actual and Predicted from CNN-Ridge',fontsize = 15)
    plt.savefig(produces["cnn_fit"], bbox_inches='tight')
    plt.clf()
    pass
""" After training CNN, use the trained model to extract image feature vectors in the validation set."""

import numpy as np
import pandas as pd
import torch
import torchvision 
from torchvision import datasets, models, transforms
from tqdm import tqdm
import os

        
def feature_extract(VAL_DIR,cnn_model,image_locs_bin):

    # Load image files dataset to later merge features with features
    df_images = image_locs_bin

    # Load model and truncate it to just before classification step
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(cnn_model)
    model.classifier = model.classifier[:4]  # truncation step
    model = model.to(device)
    model.eval()    # set to eval mode
    
    # Transformation of data
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create validation datasets
    val_data = datasets.ImageFolder(VAL_DIR,data_transform)

    # Load validation dataloaders
    dataloaders = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False, num_workers=4) 
    
    # Placeholder for batch features
    PREDS = []
    
    for inputs, idx in tqdm(dataloaders):
        
        # move to device
        inputs = inputs.to(device)

        # forward pass [with feature extraction]
        preds = model(inputs)

        # add  preds to lists
        PREDS.append(preds.detach().cpu().numpy())

    PREDS = np.concatenate(PREDS)
    
    # match image features w/ image file names
    file_names = []
    for i in range(len(val_data.imgs)):
        file_names.append(os.path.basename(val_data.imgs[i][0]))
        
    # Prepare to aggregate feature data with household cluster data.
    preds_df = pd.DataFrame(PREDS)
    preds_df = preds_df.add_prefix('feature_')
    preds_df["image_name"] = file_names
    df_features = pd.merge(left=df_images, right=preds_df, on='image_name')
    
    hh_clusters = df_features.groupby(['lon_modified', 'lat_modified'])
    mean_features = np.zeros((780, 4098))    # there are 780 household clusters, 4096 features + consumption + nlights mean
    
    # Each clusters data is aggregated into a single row and stored in a new dataframe
    for i,j in enumerate(hh_clusters):
        lon, lat = j[0]
        cluster_image = df_features[(df_features['lat_modified'] == lat) & 
                                    (df_features['lon_modified'] == lon)].reset_index(drop=True)
        XY = cluster_image.drop(['image_name','train','image_lat','nightlights_bin',
                        'image_lon','lat_modified','lon_modified'], axis = 1)
        mean_features[i,] = XY.mean(numeric_only=True,axis=0)

    df_mean_features = pd.DataFrame(mean_features, columns=XY.columns)
    
    return df_mean_features



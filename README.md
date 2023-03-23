# Convolutional Neural Networks and Satellite Imagery for Economic Analysis
###### Author: M.Danial Syed

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/MDanialSyed/cnn_satellite/main.svg)](https://results.pre-commit.ci/latest/github/MDanialSyed/cnn_satellite/main)
 [![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


<img src="https://github.com/MDanialSyed/cnn_satellite/blob/main/paper/high_light.png" width="260"> <img src="https://github.com/MDanialSyed/cnn_satellite/blob/main/paper/active_hl.png" width="250">

*(Activation Map of an image using a VGG_11 network, produced using [PyTorch Visualization Tool](https://github.com/utkuozbulak/pytorch-cnn-visualizations)*)

This repository contains my final project based on 'Combining satellite imagery and machine learning to predict poverty' (Jean et al., 2016) for the course Effective Programming Practices for Economists at Bonn University held during the Winter Semester 2022-2023. 

## Replication project

The key contribution of this replication project is to conduct data management, model training, and prediction of consumption expenditure in a reproducible and accessible way. The reason for this is that the original study and related works required considerable manual effort to obtain multiple data files and run several scripts to reproduce results. I ameliorate this complexity by instead drawing on `pytask` as a workflow management system, which is well-suited to a project of this scale. In doing so, this project draws on concepts from this course and demonstrates Python's rich scientific libraries in the process, thus providing a suitable platform for users to easily scale up this work in the future.

## Repository guide

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter) and the [econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates) and has the following setup:

- `data`: LSMS survey data and download destination for satellite data and VIIRS nightlights data.
- `data_management`: code to process and merge relevant data.
- `helper_codes`: auxiliary functions used in data managemenmt, model training, and prediction.
- `analysis`: analysis codes of data using machine learning and CNN.
- `final`: codes and tasks responsible for generating final plot. 
- `test`: functions to check the expected output of various project phases. 
- `paper`: compiled and create term paper.

## Usage

The following directions are for reproducing this project:

1. To get started, clone this repository and navigate to its root directory.
2. Open a terminal window and execute these commands to create and activate the project.

```console
$ conda env create
$ conda activate cnn_satellite
```

3. Now you can build the project in the activated project environment using 

```console
$ pytask
```

or 

```console
$ pytask -s 
```

to view progress bars for computationally intensive tasks. 

4. (Optional) In case you encounter segmentation faults in the previous step, please run `conda install pytorch torchvision -c pytorch` in the activated project environment. 

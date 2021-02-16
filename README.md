# Learned-MECT
This repository contains codes for manuscript: 
Inkinen et al. (2021) "Fast deep learning spectral reconstruction for multi-bin photon counting computed tomography".


## Prerequisites
This repository has been tested under following settings:
- Ubuntu 18.04
- Python 3.6.10 
  - Pytorch 1.1.0 
  -  Operator discretization library [ODL](https://odlgroup.github.io/odl/) 
- CUDA toolkit 10.2 
- Matlab R2018b (with [Astra toolbox v1.8](https://www.astra-toolbox.com/))

## Usage 
### Training and testing
The network training and testing codes are located in: ```Neural_networks/train.py ``` and ```Neural_networks/test.py ```. 
The network structure (Unet, LGD, LGD_DF) and dataset (Ellipse, MUSIC) is selected in training and testing with argument parser eg. as: ```train.py  --dataset Ellipse, --model Unet ```. 
Configuration settings intraining and testing: The default settings can be found from ```Neural_networks/config/config.yaml ``` where you can modify the parameters and settings needed for training. 

The network structures can be found under:```Neural_networks/networks```. 
The pytorch dataset classes for both datasets can be under:```Neural_networks/dataset```. 
Computed tomography geometry settings can be found under:```Neural_networks/geometries```. 
Utility functions for visualization and ```pyOperator.py``` to wrap the ODL operators to pytorch can be found under:```Neural_networks/utils```. 

Trained models are stored into: ```Neural_networks/trained_models```. 
Test outputs are stored into: ```Neural_networks/results```. 

## Datasets 
### MUSIC
The Multi-Spectral Imaging via Computed Tomography (MUSIC) dataset can be downloaded from [here](http://easi-cil.compute.dtu.dk/index.php/datasets/music/).
Preprocessing code for manual cropping and ADMM+TV reconstruction can be found from ```Data/MUSIC_data_reconstruction.py ```.

### Ellipse dataset
The ellipse dataset is generated using matlab. The codes for dataset generation can be found under  ```Simulate_ellipse_data ```.

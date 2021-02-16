# Learned-MECT
This repository contains codes for manuscript: 
Inkinen et al. (2021) "Fast deep learning spectral reconstruction for multi-bin photon counting computed tomography".



## Prerequisites
What things you need to install the software and how to install them
Scripts will need Matlab, Python, and Pytorch... version!

This repository has been tested under following settings:
- Ubuntu  16.04 CHECK
- Python 3.7 CHECK
  - Pytorch 1.1.0 CHECK
  -  Operator discretization library [ODL](https://odlgroup.github.io/odl/) 
- CUDA 9.0 or above CHECK
- Matlab R2018b (with [Astra toolbox v1.8](https://www.astra-toolbox.com/))

## Usage 
### Training and testing
Configure the training and testing: The default settings can be found from ```config/xx.yaml ``` where you can modify the parameters needed for learning. 


## Datasets 
### MUSIC
The Multi-Spectral Imaging via Computed Tomography (MUSIC) dataset can be downloaded from [here](http://easi-cil.compute.dtu.dk/index.php/datasets/music/).
Preprocessing code for manual cropping and ADMM+TV reconstruction can be found from ```data/YY ```.

### Ellipse dataset
The ellipse dataset is generated using matlab. The codes for dataset generation can be found under  ```Simulate_ellipse_data ```.


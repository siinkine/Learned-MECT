import odl
import numpy as np, h5py 
import torch
from subprocess import check_output 
import matplotlib.pyplot as plt
import astra
import os
import os.path as path
import yaml
import numpy as np


def resolve_expression(config):
    if type(config) is dict:
        new_config = {}
        for k, v in config.items():
            v = resolve_expression(v)
            new_config[k] = v
        config = new_config
    return config


def get_config(config_file):
    ''' Load in configuration file '''    
    with open(config_file) as f:
        config = resolve_expression(yaml.load(f, Loader=yaml.FullLoader))
        
    return config 


def save_lossess(full_save_name, lossVal_mean, lossTrain, psnrTrain, epochs_list):
    '''After training save loss vectors in .h5'''
    hf = h5py.File(full_save_name, 'w')
    hf.create_dataset('lossVal_mean', data = lossVal_mean)
    hf.create_dataset('lossTrain', data = lossTrain)
    hf.create_dataset('psnrTrain', data = psnrTrain)
    hf.create_dataset('epochs', data = epochs_list)
    hf.close()
    return None

    
def read_matlab(filename):
    ''''This reads in matlab files for the ellipse data'''
    def conv(path=''):
        p = path or '/'
        paths[p] = ret = {}
        for k, v in f[p].items():
            if type(v).__name__ == 'Group':
                ret[k] = conv(f'{path}/{k}')  # Nested struct
                continue
            v = v[()]  # It's a Numpy array now
            if v.dtype == 'object':
                # HDF5ObjectReferences are converted into a list of actual pointers
                ret[k] = [r and paths.get(f[r].name, f[r].name) for r in v.flat]
            else:
                # Matrices and other numeric arrays
                ret[k] = v if v.ndim < 2 else v.swapaxes(-1, -2)
        return ret

    paths = {}
    with h5py.File(filename, 'r') as f:
        return conv()


def FBP_recon_parallel_geom(sino):
    ''''This generates the FBP reconstruction for the ellipse data in parallel beam geometry Ellipse data'''
    
    vol_geom = astra.create_vol_geom(256,256)
    angles = np.linspace(0, 2*np.pi, sino.shape[0], False)
    proj_geom = astra.create_proj_geom('parallel', 1, sino.shape[1], angles)
   
    sinogram_id = astra.data2d.create('-sino', proj_geom, 0);
    astra.data2d.store(sinogram_id, sino);
    
    rec_id = astra.data2d.create('-vol', vol_geom)

    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = { 'FilterType': 'hann' }


    alg_id = astra.algorithm.create(cfg)

    astra.algorithm.run(alg_id, 1)

    # Get the result
    rec = astra.data2d.get(rec_id)
    
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    return rec, sino


def read_in_data(filename):
    '''Reads in the reconstructed data from MUSIC dataset'''
    
    hf = h5py.File(filename, 'r')
    hf.keys()
    
    data1 = hf.get('reconstructions')
    reco = np.array(data1)
    
    data2 = hf.get('sinogram')
    sino = np.array(data2)

    data3 = hf.get('crop_regions')
    crop_regions = np.array(data3)
    hf.close()   
    
    return reco, sino, crop_regions

def FBPrecon2D(sino):
    ''' FBP reconstruction for MUSIC data in 2D'''
        
    pixel_size = 0.077    # Pixel size
    SDD = 115.55          # Distance source to detector
    SAD = 57.50           # Distance between source and rotation axis
    num_angles = 37 
    
    size = 256
    det_num = 320         # Number of detector elements (shapes the magnification of the reconstruction)
    
    
    space = odl.uniform_discr([-128, -128], [128, 128], [size , size ],dtype='float32')
    
    geometry = odl.tomo.cone_beam_geometry(space, src_radius=(SDD/pixel_size), det_radius=(SDD-SAD)/pixel_size, num_angles = num_angles, det_shape=det_num)
    
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda') 
    fbp_op = odl.tomo.fbp_op(ray_trafo,filter_type='Hann',frequency_scaling=0.6)
    rec = np.zeros((sino.shape[1], sino.shape[1]))     
     
    sino_adapt = np.zeros((num_angles,det_num )) 
    sino_adapt[:, int((det_num-size )/2) :int((det_num-size )/2) +size ] = sino[:,:]
    
    X = fbp_op(sino_adapt)
    rec = X.data
      
    return rec, sino_adapt

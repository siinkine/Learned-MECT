from __future__ import print_function, division
import odl
import numpy as np
from utils.pyOperator import OperatorAsModule
import torch


def ellipse_geometry():
    """Ellipse dataset reconstruction geometry"""
    
    pixel_size = 0.077    # Pixel size
    SDD = 115.55          # Distance source to detector
    SAD = 57.50           # Distance between source and rotation axis
    num_angles = 37 
    
    size = 256
    det_num = 320         # Number of detector elements (shapes the magnification of the reconstruction)
        
    space = odl.uniform_discr([-128, -128], [128, 128], [size , size ],dtype='float32')
    
    val = -0.084277220364176883
    
    apart = odl.uniform_partition(0+val, 2*np.pi+val, num_angles)
    dpart = odl.uniform_partition(-128, 128, 256)
    
    geometry =  odl.tomo.Parallel2dGeometry(apart, dpart)

    
    ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda') 
    fbp_op = odl.tomo.fbp_op(ray_trafo,filter_type='Hann',frequency_scaling=0.6)
    
    return ray_trafo, fbp_op
    

def MUSIC_geometry(): 
    """MUSIC dataset reconstruction geometry"""
    

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
    
    return ray_trafo,fbp_op


def get_operators(dataset,dataset_train, device):
    ''' Get operators needed for the LGD network '''
    
    if dataset=='Ellipse':
    
       ray_trafo, fbp_op =  ellipse_geometry()

    if dataset=='MUSIC':
       ray_trafo, fbp_op =  MUSIC_geometry()
       
    # Make pytorch Modules from ODL operators
    fwd_op_mod = OperatorAsModule(ray_trafo).to(device)
    fwd_op_adj_mod = OperatorAsModule(ray_trafo.adjoint).to(device)
    fbp_op_mod = OperatorAsModule(fbp_op).to(device)
    
    #Calculate eta:
    sample = dataset_train[0]
    I = sample['FBP']
    I = I.numpy()
     
    img = np.zeros((1,1,I.shape[1],I.shape[2]))
    img[0,0,:,:] =  I[0,:,:]
     
    normal = fwd_op_adj_mod(fwd_op_mod(torch.from_numpy(img).float()))
    opnorm = torch.sqrt(torch.mean(normal ** 2)) / torch.sqrt(torch.mean(torch.from_numpy(img).float() ** 2))
    eta = 1 / opnorm         

    return  fwd_op_mod,fwd_op_adj_mod, fbp_op_mod, eta

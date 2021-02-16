#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code calculates the reconstructions for the MUSIC dataset using ODL library and ADMM TV.
each energy channels is reconstucted separately and number of iterations is set to 2000, and regulatization to 0.8.

"""

import numpy as np
import matplotlib.pyplot as plt
import odl
import h5py
import glob
import astra 
import scipy
import time
from numpy import *
from scipy import interpolate
from numba import jit


# --- Set-up ODL geometry  --- #

# MUSIC dataset imaging geometry:
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


#%% Functions needed for data loading and pre-processing:

def readMusicData(path_data, read_in_reconstruction = False):
    "This function reads in music data "

    path_data_sinogram = path_data+'projections/'
    path_data_ARTReco = path_data+'reconstruction/'

    #Get datanames:
    file_name_sinogram = glob.glob(path_data_sinogram + '*.h5')
    file_name_ARTReco = glob.glob(path_data_ARTReco + '*.h5')

    #Read the data:    
    hf = h5py.File(file_name_sinogram[0], 'r')
    sinogram = np.array(hf['data']['value'], order='F')
    hf.close()    
    if read_in_reconstruction:
        hf = h5py.File(file_name_ARTReco[0], 'r')
        reconstruction = np.array(hf['data']['value'], order='F')
        hf.close()    
        return sinogram, reconstruction

    return sinogram
  

@jit(nopython=True)
def interpolate1D(arr_3d):
    "This makes interpolation to image data fills nans, fast implementation"
#    https://stackoverflow.com/questions/30910944/fast-1d-linear-np-nan-interpolation-over-large-3d-array/31256007

    result = np.zeros_like(arr_3d)
    for i in range(arr_3d.shape[1]):
        for j in range(arr_3d.shape[2]):
            arr=arr_3d[:,i,j]
            # If all elements are nan then cannot conduct linear interpolation.
            if np.sum(np.isnan(arr))==arr.shape[0]:
                result[:,i,j]=arr
            else:
                # If the first elemet is nan, then assign the value of its right nearest neighbor to it.
                if np.isnan(arr[0]):
                    arr[0]=arr[~np.isnan(arr)][0]
                # If the last element is nan, then assign the value of its left nearest neighbor to it.
                if np.isnan(arr[-1]):
                    arr[-1]=arr[~np.isnan(arr)][-1]
                # If the element is in the middle and its value is nan, do linear interpolation using neighbor values.
                for k in range(arr.shape[0]):
                    if np.isnan(arr[k]):
                        x=k
                        x1=x-1
                        x2=x+1
                        # Find left neighbor whose value is not nan.
                        while x1>=0:
                            if np.isnan(arr[x1]):
                                x1=x1-1
                            else:
                                y1=arr[x1]
                                break
                        # Find right neighbor whose value is not nan.
                        while x2<arr.shape[0]:
                            if np.isnan(arr[x2]):
                                x2=x2+1
                            else:
                                y2=arr[x2]
                                break
                        # Calculate the slope and intercept determined by the left and right neighbors.
                        slope=(y2-y1)/(x2-x1)
                        intercept=y1-slope*x1
                        # Linear interpolation and assignment.
                        y=slope*x+intercept
                        arr[x]=y
                result[:,i,j]=arr
    return result


def preProcessSinograms(sinograms):
   
    "Pre-processing of the sinograms"
   
    #Find the mask for collimator first:
    arr_3d = sinograms[:,:,:,30]
    I = np.sum(arr_3d[:,:, :], 1)
    idxNaNs = isnan(I)
    I[idxNaNs] = 0
    BW  = I > 0
    BW = BW.astype('int')
    maskNaN = np.ones(I.shape)
    maskNaN[BW == 0 ] = np.nan  #Assign zeros to nan

        
    for iBin in range(sinograms.shape[3]): #Channelwise interpolation
            
        arr_3d = sinograms[:,:,:,iBin]        
        for ix in range(arr_3d.shape[1]): 
            I = arr_3d[:,ix,:]
            arr_3d[:,ix,:] = maskNaN*I #Assign mask with nans and interpolate data to nans
            
        #Interpolate            
        arr_3d = np.transpose(arr_3d,(2, 0, 1 ))
        arr_3d = interpolate1D(arr_3d)
        arr_3d = np.transpose(arr_3d,(1, 2, 0 ))
        
        sinograms[:,:,:,iBin]  = arr_3d 
        
    return sinograms


def make_ADMM_TV_recon(sino):
    "This function makes the reconstruction for one spectral channel i.e. sinogram." 
    
    
    data = fbp_op.domain.zero()  #intialize sinogram object of  discr.lp_discr.DiscreteLpElement
    data[:,:] = (sino) #Assign sinogram to data
    
    # --- Set up the inverse problem --- #
    
    # Gradient operator for the TV part
    grad = odl.Gradient(space)
    
    # Stacking of the two operators
    L = odl.BroadcastOperator(ray_trafo, grad)
    
    # Data matching and regularization functionals
    data_fit = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)
    reg_func = 0.8 * odl.solvers.L1Norm(grad.range) #regularization strength
    g = odl.solvers.SeparableSum(data_fit, reg_func)
    
    # We don't use the f functional, setting it to zero
    f = odl.solvers.ZeroFunctional(L.domain)
    
    # --- Select parameters and solve using ADMM --- #
    
    # Estimated operator norm, add 10 percent for some safety margin
    op_norm = 1.1 * odl.power_method_opnorm(L, maxiter=20)
    
    niter = 2000  # Number of iterations
    sigma = 2.0  # Step size for g.proximal
    tau = sigma / op_norm ** 2  # Step size for f.proximal
    
    ## Optionally pass a callback to the solver to display intermediate results
    #callback = (odl.solvers.CallbackPrintIteration(step=10) &
    #            odl.solvers.CallbackShow(step=10))
    
    # Choose a starting point
    x = L.domain.zero()
    
    # Run the algorithm
    odl.solvers.admm_linearized(x, f, g, L, tau, sigma, niter)
    
    #fig, ax = plt.subplots()
    #im = ax.imshow(x.data,  vmin=-0.000, vmax = 0.02)
    #cbar = fig.colorbar(im)
    #plt.show()
    #plt.gray()
    
    return x.data

 
def correct_sinogram(sino, sino_TE):
    
    "This function post-corrects the sinogram center axis. This is needed for a some of the datasets" 
    
    BW = sino_TE > 20#25
    BW_part = BW[:,10:220]    
    res_all = []
    for i in range(0, BW.shape[0]):
        profile = BW_part[i,:]
        
        res = [] 
        for idx in range(0, len(profile)) : 
            if profile[idx] > 0.4: 
                res.append(idx) 
        
        val = (np.max(res)-np.min(res))/2 + np.min(res) 
        res_all.append(val)
    x = np.round(np.median(res_all))+10
 #   BW = BW.astype("int")
 #   BW[20,x.astype("int")] = 2     
    x=122

    loc = 128
    ra = 100
    sino_corr = np.zeros(sino.shape)
    sino_corr[:,loc-ra:loc+ra ]  =sino[:,x-ra:x+ra]

    #sino[:,x.astype("int")-ra:x.astype("int")+ra]
    
    return sino_corr


def crop_sinogram(sinogram):
    "This interactive function is used to manually adjust the crop region for the sample."

    crop_regions = [100, 460]#285
    orig_size = sinogram.shape[1]
    
    answer="N"
    while(answer=="N"):    
    
        sinogram_crop = sinogram[:,crop_regions[0]:crop_regions[1],:,32:32+64]    
        fig, ax = plt.subplots()
        im = ax.imshow(sinogram_crop[30,:,:,1],  vmin=-0.000, vmax = 2)
        cbar = fig.colorbar(im)
        plt.show()
        plt.gray()
        
        
        answer = input("Manual cropping, are the values " + str(crop_regions) + " in Z-direction okay (upper limit"+ str(orig_size)+ ") [Y/N]? "  )
        
        if answer=="N":
            answer_value0 = input("Give new value for " + str(crop_regions[0])+  " : " )
            answer_value1 = input("Give new value for " + str(crop_regions[1])+  " : " )
            
            crop_regions = [int(answer_value0), int(answer_value1)];

    return sinogram_crop, crop_regions
     




def save_data(save_name, reco, sino, crop_regions, save_path):
     
    full_save_name = save_path+ save_name
    hf = h5py.File(full_save_name, 'w')
    hf.create_dataset('sinogram', data=sino)
    hf.create_dataset('reconstructions', data=reco)
    hf.create_dataset('crop_regions', data=crop_regions)
    hf.close()



#%% 
#käyty läpi: Sample_23012018, Sample_24012018 Sample_31102016

# --- Load and preprocess data  --- #
dataset_names = ["Fruits"]
#["NonThreat"] # [ "Sample_23012018"]# "Sample_24012018",  "Sample_31102016", "Threat", "Fluids", "NonThreat", "Fruits"]
#Check fruits dataset sinograms setting and NonThread has it been calculated
#NonThread has not been calculated so put that into calculation

#x=112 --> Sample_31102016

#101, 285 regions fruits

path_data_location = "/media/satu/Transcend/MUSIC_data/MUSIC3D_HDF5/"

#Assing save path
save_path = '/media/satu/Transcend/MUSIC_data/Final/Data/MUSIC/data/'


for name_data in dataset_names:
    
    path_data = path_data_location+name_data+"/fullSpectrum/"
    
    print(path_data)

    time_start = time.time()
    sinogram  = readMusicData(path_data)
    time_elapsed = (time.time() - time_start)
    print("time elapset reading in data")
    print(time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))

        
    time_start = time.time()
    sinogram = preProcessSinograms(sinogram)
    time_elapsed = (time.time() - time_start)

    print("time elapset reading in preprocessing")
    print(time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))
    
    
    sinogram, crop_regions =  crop_sinogram(sinogram)
     
    
    answer = input("do sinogram center correction? [Y/N]" )
        
        
    # --- Loop sinogram data slice by slice and channel by channel   --- #
    
    
    for z_slice in range(148, sinogram.shape[1]):
        
        time_start = time.time()
        
        print("Slice number: "+str(z_slice))

        reco = np.zeros((sinogram.shape[3], 256, 256),dtype='float32') #init reconstruction volume
    
        sino_all =  sinogram[:,z_slice,:,:]
        for energy_channel in range(sinogram.shape[3]): #loop through energy channels

            print("Energy channel: "+str(energy_channel) +"/"+str(sinogram.shape[3]))

            sino = sinogram[:,z_slice,:,energy_channel]
            
            sino_TE = np.sum(sinogram[:,z_slice,:,:], axis= 2)
            
            if answer=="Y":
            
                sino = correct_sinogram(sino, sino_TE) 
                sino_all[:,:,energy_channel] = sino
            
            #Ajust to fit detector pixels
            sino_adapt = np.zeros((num_angles,det_num )) 
            sino_adapt[:, int((det_num-size )/2) :int((det_num-size )/2) +size ] = sino
            
            rec = make_ADMM_TV_recon(sino_adapt) #reconstruct
            
            if energy_channel==34: #Show one specral channel as sanity check
                fig, ax = plt.subplots()
                im = ax.imshow(rec) #, vmin=-0.000, vmax = 0.04)
                cbar = fig.colorbar(im)
                plt.show()
                plt.gray()
            
            #Assign data
            reco[energy_channel, :,:] = rec
        
        #Save data:        
        
        save_name = name_data +'_slice_'+ str(z_slice)
        
        
        save_data(save_name,  reco, sino_all, crop_regions, save_path)
        
        time_elapsed = (time.time() - time_start)
        print("time elapset for reconstructions for all channels")
        print(time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))
    
    

    
#%%

#old
#path_data = '/media/satu/Transcend/MUSIC_data/MUSIC3D_HDF5/Sample_23012018/fullSpectrum/'
#path_data ='/media/satu/Transcend/MUSIC_data/MUSIC3D_HDF5/Sample_24012018/fullSpectrum/' 
#path_data ='/media/satu/Transcend/MUSIC_data/MUSIC3D_HDF5/Sample_31102016/fullSpectrum/' 
#path_data ='/media/satu/Transcend/MUSIC_data/MUSIC3D_HDF5/Threat/fullSpectrum/' 
#path_data ='/media/satu/Transcend/MUSIC_data/MUSIC3D_HDF5/Fluids/fullSpectrum/' 
#path_data ='/media/satu/Transcend/MUSIC_data/MUSIC3D_HDF5/Fruits/fullSpectrum/' 
#path_data ='/media/satu/Transcend/MUSIC_data/MUSIC3D_HDF5/NonThreat/fullSpectrum/'

#%%

def read_in_data(filename):
    "reads in the reconstructed dataset"
    #filename = '/media/satu/Transcend/MUSIC_data/MECT_MUSIC/Python/Reconstructions/data/Sample_23012018_slice_16'

    
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

#%%


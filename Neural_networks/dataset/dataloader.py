from __future__ import print_function, division
import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms, utils
import numpy as np, h5py 
from numpy import inf
from utils import util as util


#%% Ellipse dataset:
class EllipseDataset(Dataset):
    """ Ellipse spectral dataset random channel """
    
    def __init__(self, root_dir, low_bin = 26, high_bin = 90,  transform = None):
        """
        Parameters
        ----------
        root_dir(String): Path to the data
        low_bin: lower limit to spectral channels
        high_bin: higher limit to spectral channels
        transform(callable, optional) : Optional transforms to be applied on a sample
        (not used)
        """
        self.root_dir = root_dir
        self.low_bin = low_bin
        self.high_bin = high_bin
        self.transform = transform
        self.data_files = []
        for file in os.listdir(self.root_dir):
          if file.endswith(".mat"):
              self.data_files.append(os.path.join(self.root_dir, file))
              
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, index):
        
    
        if torch.is_tensor(index):
            index = index.tolist()
        
        data_name = os.path.join(self.root_dir,
                                self.data_files[index])
            
        t = util.read_matlab(data_name)
        X = t['X']
        y = t['y']
        size_info = t['size_info']
        
        X = np.transpose(X,(1, 2, 0))        
        X = X[:,:,self.low_bin:self.high_bin]

        
        sino = y.reshape(size_info[0][2].astype('int'), size_info[0][1].astype('int'), y.shape[1])
        sino = sino[:,:,self.low_bin:self.high_bin]
        
        #Random spectral channel:
        random_val = np.random.randint(sino.shape[2], size = 1)
        sino = sino[:,:,random_val]
       
        X = X[:,:, random_val]

        x1 = np.squeeze(X)
        x1 =  np.flipud((np.rot90(x1,2).T)) #rotations to match the pre-calculated FBP in matlab (ASTRA) to ODL output 
        X[:,:, 0] = x1 
        s = np.squeeze(np.transpose(sino/size_info[0][0]))
        
        r,sino_adapt = util.FBP_recon_parallel_geom(s)        
        r =  np.flipud((np.rot90(r,2).T))
        rec  = r[:,:, None]

        sample = {'FBP': rec, 'GT': X, 'sino': sino_adapt}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
        
class EllipseAllBinsDataset(Dataset):
    """ Ellipse spectral dataset all channels """
    
    def __init__(self, root_dir, low_bin = 26, high_bin = 90,  transform = None):
        """
        Parameters
        ----------
        root_dir(String): Path to the data
        low_bin: lower limit to spectral channels
        high_bin: higher limit to spectral channels
        transform(callable, optional) : Optional transforms to be applied on a sample
        (not used)
        """
        self.root_dir = root_dir
        self.low_bin = low_bin
        self.high_bin = high_bin
        self.transform = transform
        self.data_files = []
        for file in os.listdir(self.root_dir):
          if file.endswith(".mat"):
              self.data_files.append(os.path.join(self.root_dir, file))
              
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, index):
        
    
        if torch.is_tensor(index):
            index = index.tolist()
            
        data_name = os.path.join(self.root_dir,
                                self.data_files[index])
         
        idx = data_name.rfind("data") #get name
        name = data_name[idx:]
        name = name[0:len(name)-4] 
       
        t = util.read_matlab(data_name)
        X = t['X']
        y = t['y']
        size_info = t['size_info']
        
        X = np.transpose(X,(1, 2, 0))        
        X = X[:,:,self.low_bin:self.high_bin]
        
        sino = y.reshape(size_info[0][2].astype('int'),size_info[0][1].astype('int'),y.shape[1])
        sino = sino[:,:,self.low_bin:self.high_bin]
        
        
        for i in range(sino.shape[2]): #loop spectral channels
            
            sino_part = sino[:,:,i]
            s = np.squeeze(np.transpose(sino_part/size_info[0][0]))
            r,sino_adapt = util.FBP_recon_parallel_geom(s) 
            
            X_part = X[:,:, i]
            x1 = np.squeeze(X_part)
            x1 =  np.flipud((np.rot90(x1,2).T))
        
            
            if i == 0: #init on first round
                rec_all = np.zeros((r.shape[0], r.shape[1], sino.shape[2]))
                sino_all = np.zeros((sino_adapt.shape[0], sino_adapt.shape[1], sino.shape[2]))
                X2 = np.zeros((r.shape[0], r.shape[1], sino.shape[2]))  
  
            r = np.nan_to_num(r)
            rec_all[:,:, i] = r
            sino_all[:,:, i] = sino_adapt
            X2[:,:, i] = x1

        sample = {'FBP': rec_all, 'GT': X2, 'sino': sino_all, 'name': name}
    
        return sample



class RandomChannel(object):
        """Picks one channel randomly"""
        def __call__(self, sample):
            
            FBP, GT = sample['FBP'], sample['GT']
            
            random_val = np.random.randint(FBP.shape[2], size=1)
            FBP = FBP[:,:, random_val]
         
            GT = GT[:,:, random_val]
            sample = {'FBP': FBP, 'GT': GT}
            
            return sample

        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors. Also checks for infs """
        
    def __call__(self, sample):
        
        FBP, GT, sino = sample['FBP'], sample['GT'], sample['sino']
 
        FBP = FBP.transpose((2, 0, 1))
        FBP [FBP  == -inf] = 0
        #  FBP = np.sum(FBP, axis = 0, keepdims=True) # this makes total energy images
        
        GT = GT.transpose((2, 0, 1))
        GT [GT  == -inf] = 0
      #  GT = np.sum(GT, axis = 0, keepdims=True)        
        
        sino2   = np.zeros((1,sino.shape[0],sino.shape[1]))
        sino2[0,:,:] = sino  
      
        return {'FBP': torch.from_numpy(np.flip(FBP,axis=0).copy()).float(),
                'GT': torch.from_numpy(GT).float(), 
                'sino': torch.from_numpy(sino2).float()}

    
#%% MUSIC dataset:
class MUSICDataset(Dataset):
        """ Spectral MUSIC dataset random channel"""
        
        def __init__(self, root_dir, transform = None):
            """
            Parameters
            ----------
            root_dir(String): Path to the data
            transform(callable, optional) : Optional transforms to be applied on a sample
            (not used)
            """
            self.root_dir = root_dir
            self.transform = transform
            self.data_files  = [f for f in os.listdir(self.root_dir)]
             
        def __len__(self):
            return len(self.data_files)
        
        
        def __getitem__(self, index):
        
            if torch.is_tensor(index):
                index = index.tolist()
            
            data_name = os.path.join(self.root_dir,
                                    self.data_files[index])
                
            GT, sino, crop_regions = util.read_in_data(data_name)
                        
            #Random spectral channel:
            random_val = np.random.randint(sino.shape[2], size = 1)
            
            sino_part = sino[:,:,random_val]
            rec, sino_adapt = util.FBPrecon2D(np.squeeze(sino_part))
            
            rec = np.nan_to_num(rec)
            GT = np.nan_to_num(GT)
        
            rec = rec[:,:,None]
            
            GT = np.squeeze(GT[random_val,:,:])
            GT = GT[:,:,None]

            sample = {'FBP': rec, 'GT': GT,  'sino': sino_adapt}
            
            if self.transform:
                sample = self.transform(sample)
            
            return sample


class MUSICAllBinsDataset(Dataset):
        """ Spectral MUSIC dataset all channels """
        
        def __init__(self, root_dir, transform = None):
            """
            Parameters
            ----------
            root_dir(String): Path to the data
            transform(callable, optional) : Optional transforms to be applied on a sample
            (not used)
            """
            self.root_dir = root_dir
            self.transform = transform
            self.data_files  = [f for f in os.listdir(self.root_dir)]
            
            #Re-order:
            ind = self.data_files[0].find('slice')
            name_header = self.data_files[0][0:ind+6]
            for slice_number in range(len(self.data_files)):
                self.data_files[slice_number] = name_header + str(slice_number)
            
             
        def __len__(self):
            return len(self.data_files)
        
        
        def __getitem__(self, index):
        
    
            if torch.is_tensor(index):
                index = index.tolist()
            
            data_name = os.path.join(self.root_dir,
                                    self.data_files[index])
                
            GT, sino, crop_regions = util.read_in_data(data_name)
            GT = np.nan_to_num(GT)
            GT = GT.transpose((1, 2, 0))

                 
            for i in range(sino.shape[2]): #loop spectral channels
                sino_part = sino[:,:,i]
                rec, sino_adapt = util.FBPrecon2D(np.squeeze(sino_part))
                
                if i == 0: #init on first round
                    rec_all = np.zeros((rec.shape[0], rec.shape[1], sino.shape[2]))
                    sino_all = np.zeros((sino_adapt.shape[0], sino_adapt.shape[1], sino.shape[2]))
    
                 
                rec = np.nan_to_num(rec)
                
                rec_all[:,:, i] = rec
                sino_all[:,:, i] = sino_adapt
                

            sample = {'FBP': rec_all, 'GT': GT,  'sino': sino_all, 'name': self.data_files[index]}
            
            if self.transform:
                sample = self.transform(sample)
            
            return sample



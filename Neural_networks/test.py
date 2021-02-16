from __future__ import print_function, division
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np, h5py
import dataset.dataloader as dataloader
from networks import Unet, LGD, LGD_DF 
import time
import matplotlib.pyplot as plt 
from utils.util import get_config, save_lossess
import argparse
from geometries.geometry import get_operators
import os


# Parse command line options:
parser = argparse.ArgumentParser(description = "Test UNet or LGD on MUSIC or Ellipse dataset")
parser.add_argument("--config", default = "config/config.yaml", help = "default configs")
parser.add_argument("--dataset", default = "MUSIC", help = "Dataset selection: Ellipse, MUSIC")
parser.add_argument("--model", default = "LGD_DF", help = "Model selection: Unet, LGD, LGD_DF")
args = parser.parse_args()

opts = get_config(args.config)
opts_general = opts['general']
device = opts_general['device']
iter4Net = opts_general['iter4Net'] 

cwd = os.getcwd() #current directory
os.chdir(os.path.pardir)
rwd = os.getcwd() #dir above
os.chdir(cwd)


# Dataset and dataloader:
if args.dataset == 'MUSIC':

    opts_data = opts['MUSICdataset']
    
    dataset_test =  dataloader.MUSICAllBinsDataset(root_dir = os.path.join(rwd, opts_data['path_test']))
    
    dataset_train =  dataloader.MUSICDataset(root_dir = os.path.join(rwd, opts_data['path_train']), 
                                        transform = transforms.Compose([dataloader.ToTensor()]))


elif args.dataset =='Ellipse':

    opts_data = opts['Ellipsedataset']

    dataset_test =  dataloader.EllipseAllBinsDataset(root_dir = os.path.join(rwd, opts_data['path_test']), 
                                        transform = transforms.Compose([dataloader.ToTensor()]))
    
    dataset_train =  dataloader.EllipseDataset(root_dir = os.path.join(rwd, opts_data['path_train']), 
                                        transform = transforms.Compose([dataloader.ToTensor()]))
  

dataloader_test = DataLoader(dataset_test, batch_size = 1, shuffle = True)

# Model selection:
if args.model == 'Unet':
    model = Unet.UNet(n_in = 1, n_out = 1).to(device)
    loss_train = nn.MSELoss()
    loss_val = nn.MSELoss()

elif args.model == 'LGD':
    
    fwd_op_mod,fwd_op_adj_mod, fbp_op_mod, eta = get_operators(args.dataset, dataset_train, device)
    
    model = LGD.IterativeNetwork(niter= iter4Net,
                            op = fwd_op_mod, 
                            op_adj = fwd_op_adj_mod,
                            init_op = fbp_op_mod,
                            eta = eta, 
                            loss = nn.MSELoss()).to(device)
    
elif args.model == 'LGD_DF':
    alpha = opts_general['alpha']
    fwd_op_mod,fwd_op_adj_mod, fbp_op_mod, eta = get_operators(args.dataset, dataset_train, device)
    model = LGD_DF.IterativeNetwork(niter = iter4Net,
                        op = fwd_op_mod, 
                        op_adj = fwd_op_adj_mod,
                        init_op = fbp_op_mod,
                        eta = eta,
                        alpha =alpha,
                        loss  = nn.MSELoss()).to(device)


# Load model:
path_networks = os.path.join(cwd, opts_general['path_networks'])
save_filename  = 'LGD_DF_MUSIC_1e-1.pt'#args.model+'_' +args.dataset + '.pt'
model_save_filename  = os.path.join(path_networks,  save_filename)
 
model.load_state_dict(torch.load(model_save_filename))

# Data save path: 
data_save_path = os.path.join(cwd, opts_data['path_result'],args.model, 'data/')



# ----  functions ----
def save_data(output, FBP, GT, sino, save_path_fuĺl):
    hf = h5py.File(save_path_fuĺl, 'w')
    hf.create_dataset('output', data = output)
    hf.create_dataset('FBP', data = FBP)
    hf.create_dataset('GT', data = GT)
    hf.create_dataset('sino', data = sino)
    hf.close()
    

def test_Unet(dataloader_test, model, data_save_path):
        
    model.eval()
    
    for i in  range(len(dataset_test)): #this loops samplewise
        sample = dataset_test[i]
        FBP, GT, name, sino = sample['FBP'], sample['GT'], sample['name'], sample['sino']
    
        time_start = time.time()
    
        output = np.zeros(FBP.shape)
        for iBin  in range(sino.shape[2]): # Loop throuh slices and apply the network each spectral channel
            
            #change to tensor dimensions:
            FBP_bin  = np.zeros((1,1, FBP.shape[0], FBP.shape[1]))
            GT_bin  = np.zeros((1,1, FBP.shape[0], FBP.shape[1]))
            
            sino_bin  =  np.zeros((1,1, sino.shape[0], sino.shape[1]))
                
            FBP_bin[0,0,:,:] = FBP[:,:, iBin]
            sino_bin[0,0,:,:] = sino[:,:, iBin]
            GT_bin[0,0,:,:]  = GT[:,:, iBin]
            GT_bin[GT_bin  == -np.inf] = 0
            FBP_bin[FBP_bin  == -np.inf] = 0
    
            # to torch:
            FBP_bin = torch.from_numpy(FBP_bin).float().to(device)
            sino_bin = torch.from_numpy(sino_bin).float().to(device)
            GT_bin = torch.from_numpy(GT_bin).float().to(device)
        
            output_bin  = model(FBP_bin)
                    
            #assign data
            output[:,:,iBin] = output_bin[0,0,:,:].cpu().detach()
    
        time_elapsed = (time.time() - time_start)
    
        print('Slice number: ' + str(i)+'/'+str(len(dataset_test)) )
        print("Time elapsed:")
        print(time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))
           
        #Save data:
        save_path_fuĺl = os.path.join(data_save_path,  name+'.h5')
        save_data(output, FBP, GT, sino, save_path_fuĺl)
        
    return None
    
 

def test_LGDs(dataloader_test, model, data_save_path):

    model.eval()
    
    for i in  range(len(dataset_test)): #this loops samplewise
        sample = dataset_test[i]
        FBP, GT, name, sino = sample['FBP'], sample['GT'], sample['name'], sample['sino']
    
        time_start = time.time()
    
        output = np.zeros(FBP.shape)
        for iBin  in range(sino.shape[2]): #Loop throuh slices and apply the network each spectral channel
            
            #change to tensor dimensions:
            FBP_bin  = np.zeros((1,1, FBP.shape[0], FBP.shape[1]))
            GT_bin  = np.zeros((1,1, FBP.shape[0], FBP.shape[1]))
            
            sino_bin  =  np.zeros((1,1, sino.shape[0], sino.shape[1]))
    
                
            FBP_bin[0,0,:,:] = FBP[:,:, iBin]
            sino_bin[0,0,:,:] = sino[:,:, iBin]
            GT_bin[0,0,:,:]  = GT[:,:, iBin]
            GT_bin[GT_bin  == -np.inf] = 0
            FBP_bin[FBP_bin  == -np.inf] = 0
    
            # to torch:
            FBP_bin = torch.from_numpy(FBP_bin).float().to(device)
            sino_bin = torch.from_numpy(sino_bin).float().to(device)
            GT_bin = torch.from_numpy(GT_bin).float().to(device)
            
            output_bin, loss_val = model(sino_bin, GT_bin, 0)
            
            #assign data
            output[:,:,iBin] = output_bin[0,0,:,:].cpu().detach()
    
        time_elapsed = (time.time() - time_start)
    
        print('Slice number: ' + str(i)+'/'+str(len(dataset_test)) )
        print("Time elapsed:")
        print(time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))
           
        #Save data:
        save_path_fuĺl = data_save_path + name+'.h5'
        save_data(output, FBP, GT, sino, save_path_fuĺl)

    return None

        

#  Testing:

if args.model == 'Unet':
    print('Unet')
    test_Unet(dataloader_test, model, data_save_path)
else:
    print('LGD')
    test_LGDs(dataloader_test, model, data_save_path)
    

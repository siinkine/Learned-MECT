from __future__ import print_function, division
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import dataset.dataloader as dataloader
from networks import Unet, LGD, LGD_DF 
import time
import matplotlib.pyplot as plt 
from utils.util import get_config, save_lossess 
from utils.visualizations import plot_summary, plot_progress
import argparse
from geometries.geometry import get_operators
import os


# Parse command line options:
parser = argparse.ArgumentParser(description = "Train UNet or LGD on MUSIC or Ellipse dataset")
parser.add_argument("--config", default = "config/config.yaml", help = "default configs")
parser.add_argument("--dataset", default = "Ellipse", help = "Dataset selection: Ellipse, MUSIC")
parser.add_argument("--model", default = "LGD_DF", help = "Model selection: Unet, LGD, LGD_DF")
args = parser.parse_args()

opts = get_config(args.config)
opts_general = opts['general']
print(opts_general['alpha'])
device = opts_general['device']
lr =  opts_general['lr']
epochs = opts_general['epochs']
iter4Net = opts_general['iter4Net'] 
batch_size = opts_general['batch_size'] 
seed = opts_general['seed']


cwd = os.getcwd() #current directory
os.chdir(os.path.pardir)
rwd = os.getcwd() #dir above
os.chdir(cwd)

# Dataset and dataloader:
if args.dataset == 'MUSIC':

    opts_data = opts['MUSICdataset']
    dataset_train =  dataloader.MUSICDataset(root_dir = os.path.join(rwd, opts_data['path_train']), 
                                        transform = transforms.Compose([dataloader.ToTensor()]) )
    dataset_val =  dataloader.MUSICDataset(root_dir = os.path.join(rwd, opts_data['path_val']) , 
                                        transform = transforms.Compose([dataloader.ToTensor()]))
    
elif args.dataset =='Ellipse':

    opts_data = opts['Ellipsedataset']
    dataset_train =  dataloader.EllipseDataset(root_dir =  os.path.join(rwd, opts_data['path_train']), 
                                        transform = transforms.Compose([dataloader.ToTensor()]))
    dataset_val =  dataloader.EllipseDataset(root_dir = os.path.join(rwd, opts_data['path_val']), 
                                        transform = transforms.Compose([dataloader.ToTensor()]))
  

dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, worker_init_fn=seed, num_workers=0)
dataloader_val = DataLoader(dataset_val,batch_size = 1, shuffle = True, worker_init_fn=seed, num_workers=0)  


# Model, optimizer, scheduler:
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
    #import pdb; pdb.set_trace()
    alpha = opts_general['alpha']
    fwd_op_mod,fwd_op_adj_mod, fbp_op_mod, eta = get_operators(args.dataset, dataset_train, device)
    model = LGD_DF.IterativeNetwork(niter = iter4Net,
                        op = fwd_op_mod, 
                        op_adj = fwd_op_adj_mod,
                        init_op = fbp_op_mod,
                        eta = eta,
                        alpha= alpha,
                        loss = nn.MSELoss()).to(device)



#set seed:
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



optimizer = optim.Adam(model.parameters(), lr = lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
#Save directories:
path_networks = os.path.join(cwd, opts_general['path_networks'])
save_filename  = args.model+'_' +args.dataset + '.pt'
model_save_filename  = os.path.join(path_networks,  save_filename)

save_loss_name = 'train_results.h5'
loss_save_filename = os.path.join(cwd, opts_data['path_result'],args.model, save_loss_name)

# ---functions ---
 
def train_Unet(dataloader_train, dataloader_val, model, optimizer, scheduler, loss_train,loss_val, epochs,model_save_filename, loss_save_filename):
    
    lossVal_mean = []
    lossVal = []
    lossTrain = []
    epochs_list = []
    psnrTrain  = []
    for epoch in range(epochs):  
        epochs_list.append(epoch)
        print("Epoch: " +str(epoch))    
        
        #Training:
        model.train()
        running_loss = 0.0
        running_loss_all = []
        for i, data in enumerate(dataloader_train, 0):

            FBP, GT = data['FBP'], data['GT']
            FBP = FBP.to(device, dtype=torch.float)
            GT = GT.to(device,  dtype=torch.float)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(FBP)
            loss = loss_train(outputs, GT)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_loss_all.append(loss.item())
            
            torch.cuda.empty_cache()
            if i < len(dataloader_train)-2:
                del outputs
                del FBP
                del GT
        
            if i % 10 == 9:
                #import pdb;pdb.set_trace()
               # FBP, GT = data['FBP'], data['GT']
               # plt.imshow(FBP[0,0,:,:]), plt.show()
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        
        lossTrain.append(np.mean(running_loss_all))
        l, p = plot_summary(outputs, FBP, GT, loss, epoch)
        psnrTrain.append(p)
    
        del outputs
        del FBP
        del GT
        
        lossVal = []
        #Validation:
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(dataloader_val, 0):
                FBP, GT = data['FBP'], data['GT']
                
                FBP = FBP.to(device, dtype=torch.float)
                GT = GT.to(device,  dtype=torch.float)
                outputs = model(FBP)
                val=loss_val(outputs, GT)
                lossVal.append( val.cpu().detach().numpy())
        
                torch.cuda.empty_cache()
                if i < len(dataloader_val)-2:
                    del outputs
                    del FBP
                    del GT

            del outputs
            del FBP
            del GT
        
        lossVal_mean.append(np.mean(lossVal))         
        scheduler.step()
        
        fig, ax = plt.subplots() #plot after each epoch
        ax.plot(epochs_list, lossTrain, color = 'blue', label = 'train')
        ax.plot(epochs_list, lossVal_mean, color = 'black', label = 'val')
        ax.legend()
        plt.show()
    
    
    #after training save model and loss    
    torch.save(model.state_dict(), model_save_filename)
    save_lossess(loss_save_filename, lossVal_mean, lossTrain, psnrTrain, epochs_list)

    return None

def train_LGDs(dataloader_train, dataloader_val, model, optimizer, scheduler, epochs,model_save_filename, loss_save_filename):
    
        
    lossVal = []
    lossVal_mean = []
    lossTrain = []
    psnrTrain = []
    epochs_list = []
    for epoch in range(epochs):  
    
        print("Epoch: " +str(epoch))    
        #Training:
        model.train()
        running_loss = 0.0
        running_loss_all = []
        for i, data in  enumerate(dataloader_train, 0):

            # get the inputs; data is a list of [inputs, labels]
            FBP, GT, sino = data['FBP'], data['GT'], data['sino']
    
            FBP = FBP.to(device, dtype=torch.float)
            GT = GT.to(device, dtype=torch.float)
            sino = sino.to(device, dtype=torch.float)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            
            # forward + backward + optimize        
            outputs,loss = model(sino, GT, epoch)
                       
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            running_loss_all.append(loss.item())
    
            if i % 10 == 9:
                #import pdb;pdb.set_trace()
              #  FBP, GT = data['FBP'], data['GT']
               # plt.imshow(FBP[0,0,:,:]), plt.show()
                print('[%d, %5d] loss: %.9f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
    
        lossTrain.append(np.mean(running_loss_all))
        
        l, p = plot_summary(outputs, FBP, GT, loss, epoch)
        psnrTrain.append(p)
        epochs_list.append(epoch)
        plot_progress(lossTrain, psnrTrain, epochs_list)
        

        #Validation:
        lossVal = []
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(dataloader_val, 0):
                FBP, GT, sino = data['FBP'], data['GT'], data['sino']
                
                FBP = FBP.to(device, dtype = torch.float)
                GT = GT.to(device,  dtype = torch.float)
                sino =  sino.to(device,  dtype = torch.float)
               
                outputs, loss_val = model(sino, GT, epoch)
    
            
                lossVal.append(loss_val.cpu().detach().numpy())
    
        lossVal_mean.append(np.mean(lossVal))
        scheduler.step()
    
        fig, ax = plt.subplots()
        ax.plot(epochs_list, lossTrain, color = 'blue', label ='train')
        ax.plot(epochs_list, lossVal_mean, color = 'black', label = 'val')
        ax.legend()
        plt.show()
    
    #after training save model and loss    
    torch.save(model.state_dict(), model_save_filename)
    save_lossess(loss_save_filename, lossVal_mean, lossTrain, psnrTrain, epochs_list)

    return None

 
#  Training:
time_start = time.time()

if args.model == 'Unet':
    print('Unet')
    train_Unet(dataloader_train, dataloader_val, model, optimizer, scheduler, loss_train,loss_val, epochs,model_save_filename, loss_save_filename)
else:
    print('LGD')
    train_LGDs(dataloader_train, dataloader_val, model, optimizer, scheduler, epochs,model_save_filename, loss_save_filename)
        

time_elapsed = (time.time() - time_start)

print('Finished Training')
print("Time elapsed:")
print(time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))



    

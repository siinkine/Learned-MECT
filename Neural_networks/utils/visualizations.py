from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import torch


def show_sinograms(FBP, GT):
    """Show image with landmarks"""
    plt.subplot(211)
    plt.imshow(FBP[0,:,:])
    plt.subplot(212)
    plt.imshow(GT[0,:,:])
    plt.pause(0.001)  
    plt.figure()
    plt.show()
    
    
def show_sinograms_batch(sample_batched):
    """Show metal and nometal sinograms for a batch of samples."""
    metal_sinograms_batch, nometal_sinograms_batch = \
            sample_batched['FBP'], sample_batched['GT']
    batch_size = len(metal_sinograms_batch)
    im_size = metal_sinograms_batch.size(2)
    grid_border_size = 2

    fig = plt.figure()
    for idx in range(batch_size):
        print(batch_size)
        img1 = metal_sinograms_batch[idx,0,:,:]
        img2 = nometal_sinograms_batch[idx,0,:,:]
        
        ax = fig.add_subplot(2, batch_size, 2*idx+1)
        imgplot = ax.imshow(img1, cmap="gray")
        plt.colorbar(imgplot)
        plt.axis('off')
        ax = fig.add_subplot(2, batch_size, 2*idx+2)
        imgplot2 = ax.imshow(img2, cmap="gray")
        plt.colorbar(imgplot2)
        plt.axis('off')

    
def plot_summary(result, fbp, true, loss, it):
    ''' Plots summary images during training  '''
    residual = result[0,0,:,:] - true[0,0,:,:]
    squared_error = residual ** 2
    mse = torch.mean(squared_error)
    maxval = torch.max(true[0,0,:,:]) - torch.min(true[0,0,:,:])
    psnr = 20 * torch.log10(maxval) - 10 * torch.log10(mse)
    

    l = (loss.cpu().detach().numpy())    
    p = psnr.cpu().detach().numpy()

    text = ['TRAIN Iter:' , it, ' Loss: ', "{:.4f}".format(l), ', PSNR: ', "{:.1f}".format(p)]
    #Plotting:
    cmap = "gray"    
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(10, 4))
    
    
    fig.suptitle(text, fontsize=16)
    axs[0].set_title('True')
    data=np.squeeze(true.cpu().detach().numpy())
    if np.ndim(data)==2:
         axs[0].imshow(data, cmap = cmap)
    else:        
        axs[0].imshow(np.squeeze(data[0,:,:]), cmap = cmap)
    
    axs[1].set_title('Output')
    data = np.squeeze(result.cpu().detach().numpy())
    if np.ndim(data)==2:
         axs[1].imshow(data, cmap = cmap)
    else:        
        axs[1].imshow(np.squeeze(data[0,:,:]), cmap = cmap)
    

    axs[2].set_title('FBP')
    data=np.squeeze(fbp.cpu().detach().numpy())
    if np.ndim(data)==2:
         axs[2].imshow(data, cmap = cmap)
    else:        
        axs[2].imshow(np.squeeze(data[0,:,:]), cmap = cmap)
    
    
    plt.show()
    
    return l, p

def plot_progress(l, p, x):

    plt.subplot(211)
    plt.plot(x,l)
    plt.xlabel("Iter")
    plt.ylabel("Loss")
        
    plt.subplot(212)
    plt.plot(x,p)
    plt.xlabel("Iter")
    plt.ylabel("PSNR")

    plt.show()
    
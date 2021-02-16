import torch
from torch import nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
       nn.Conv2d(in_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),   
 #      nn.GroupNorm(num_groups=4, num_channels=out_channels),
       nn.ReLU(inplace=True),
       nn.Conv2d(out_channels, out_channels, 3, padding=1),
       nn.BatchNorm2d(out_channels),   
 #      nn.GroupNorm(num_groups=4, num_channels=out_channels),
       nn.ReLU(inplace=True) )


class Iteration(nn.Module):
    def __init__(self, op, op_adj, eta):
        super().__init__()
        self.eta = eta
        self.op = op
        self.op_adj = op_adj
        
        self.dconv_down1 = double_conv(2, 16)
        self.dconv_down2 = double_conv(16, 32)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.xUp1  = nn.ConvTranspose2d(32,16,2,stride=2,padding=0)
        
        self.dconv_up1 = double_conv(32, 16)
        self.conv_last = nn.Conv2d(16, 1, 1)
        
        self.stepsize = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, cur, y):
        # Set gradient of (1/2) ||A(x) - y||^2
        grad = self.op_adj( self.op(cur) - y)
        # Combine the two with learnable parameters
        dx = torch.cat([cur, self.eta * grad], dim=1)        

        ''' mini Unet'''
        conv1 = self.dconv_down1(dx)
        dx = self.maxpool(conv1)
        conv2 = self.dconv_down2(dx)
        dx = self.xUp1(conv2)        
        
        dx = torch.cat([dx, conv1], dim=1)         
        dx = self.dconv_up1(dx)
        dx = self.conv_last(dx)
        
        # Iteration update
        return  cur + self.stepsize * dx
    #cur - 0.5 * self.eta * grad + self.stepsize * dx
    
    
class IterativeNetwork(nn.Module):
    def __init__(self, niter, op, op_adj,init_op, loss, eta):
        super().__init__()
        self.niter = niter
        self.loss = loss
        self.init_op = init_op
        
        for i in range(niter):
            iteration = Iteration(op=op, op_adj = op_adj, eta = eta)
            setattr(self, 'iteration_{}'.format(i), iteration)

    def forward(self, y, true, it, writer=None):
#        current = torch.zeros_like(true)
        current = self.init_op(y)
        
        for i in range(self.niter):
            iteration = getattr(self, 'iteration_{}'.format(i))
            
            current = iteration(current, y)
            
            #if writer:
              #  util.summary_image(writer, 'iteration_{}'.format(i), current, it)
            
        return current, self.loss(current, true)
    

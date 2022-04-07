import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class _cross_entropy2d(nn.Module):
    def __init__(self, weight=None, reduction='elementwise_mean'):
        super(_cross_entropy2d,self).__init__()
        self.weight=weight
        self.reduction = reduction
        
    def forward (self,input, target):
        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
##        print("Target labels Size:", target.size())
##        print("Network output Size:", input.size())
        loss = F.cross_entropy(input, target, 
                           weight=self.weight, 
                           reduction=self.reduction,
                           ignore_index=255)
        return loss

class cross_entropy2d(nn.Module):
    def __init__(self,size_average=True, scale_weight=None):
        super(cross_entropy2d,self).__init__()
        self.size_average = size_average
        self.scale_weight = scale_weight
        self.loss_calc = _cross_entropy2d()
        
    def forward(self, input, target):
        if self.scale_weight == None:
            n_inp = len(input)
##            print("Length of Input:",n_inp)
##            print("x_final shape:", input[0].size())
##            print("x_lid_dec shape:", input[1].size())
            scale_weight = [0.4, 1] #Weights for camera branch loss and lidar branch loss

        losses = [0.0 for _ in range(len(input))]
        loss = 0.0
        for i, inp in enumerate(input):
            cur_loss = scale_weight[i]*self.loss_calc(input=input[i],target=target) #input are the lidar branch output and image branch output

            loss = loss + cur_loss     
        return loss
    


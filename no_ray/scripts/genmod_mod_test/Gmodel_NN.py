# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:05:27 2022

@author: Jothi Thondiraj, MS student, University of Colorado Boulder.
"""
import torch
import torch.nn as nn
from torch import sigmoid
from torch import relu
#%%Configurable neural networks for the tuning:    
##FIXME Consider adding the activation functions in the initialization step itself: 
class GenNN(nn.Module):    
    def __init__(self,Layers,p_d=0):
        super(GenNN, self).__init__()
        #torch.manual_seed(42)
        self.hidden = nn.ModuleList()
        self.Drop = nn.Dropout(p=p_d)
        for l1,l2 in zip(Layers,Layers[1:]):
            self.hidden.append(nn.Linear(l1,l2))
    def forward(self,alpha,avtn_lst,it_ind):
        activation  = torch.clone(alpha)
        L = len(self.hidden)
        for k,lin_map in zip(range(L),self.hidden):
            if k < L-1:
                if avtn_lst[k] == 'None': 
                    activation = lin_map(activation)        
                    activation = self.Drop(activation) 
                else:
                    activation = avtn_lst[k](lin_map(activation))        
                    activation = self.Drop(activation) 
            else:
                if avtn_lst[k] == 'expdec': 
                    activation = torch.exp(-lin_map(activation))        
                else:
                    activation = avtn_lst[k](lin_map(activation))        

        return activation

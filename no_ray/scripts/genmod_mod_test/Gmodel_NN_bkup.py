# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:05:27 2022

@author: jothi
"""
import torch
import torch.nn as nn
#Configurable neural networks for the tuning:    
##FIXME Consider adding the activation functions in the initialization step itself: 
class GenNN(nn.Module):
    
    def __init__(self,Layers):
        super(GenNN, self).__init__()
        self.hidden = nn.ModuleList()
        for l1,l2 in zip(Layers,Layers[1:]):
            self.hidden.append(nn.Linear(l1,l2))
        #print('Layers',Layers)    
    def forward(self,alpha,avtn_lst,it_ind):
        activation  = torch.clone(alpha)
        L = len(self.hidden)
        #print('L',L)
        for k,lin_map in zip(range(L),self.hidden):
            if k < L-1:
                if avtn_lst[k] == 'None': 
                    activation =lin_map(activation)        
                else:
                    activation = avtn_lst[k](lin_map(activation))        
            else:
                    activation = torch.exp(-lin_map(activation))        
                    #activation = torch.abs(lin_map(activation))        

        return activation


# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:05:27 2022

@author: jothi
"""
import torch
import torch.nn as nn
from torch import sigmoid
from torch import relu
# from torch.nn import GELU
#%% Single layer:
# class GenNN(nn.Module):
    
#     def __init__(self,d_in,d_out):
#         super(GenNN, self).__init__()
#         self.lin1 = nn.Linear(d_in,d_out)
    
#     def forward(self,alpha):
#         self.a1 = self.lin1(alpha)
#         chat = sigmoid(self.a1)
#         return chat
#%% Double layer:
class GenNN(nn.Module):
    
    def __init__(self,d_in,H,d_out):
        super(GenNN, self).__init__()
        self.lin1 = nn.Linear(d_in,H)
        self.lin2 = nn.Linear(H,d_out)
    def forward(self,alpha,it_ind):
        # gel = nn.GELU()
        self.a1 = self.lin1(alpha)
        # self.Rl1 = torch.exp(-(self.a1)**2)
        # a1_n = self.a1
#        self.Rl1 = nn.Sigmoid()(self.a1)
        self.Rl1 = (self.a1)
        y_hid = self.lin2(self.Rl1)
        chat    = torch.exp(-y_hid)
#        if it_ind==4:
#            self.Rl1 = sigmoid(self.a1)
#            y_hid = self.lin2(self.Rl1)
#            chat    = torch.exp(-y_hid)
        # print('y1-b:',self.a1)
        # print('y1:',torch.t(self.Rl1))
        # print('y2:',torch.t(y_hid))
        # print('c_hat:',torch.t(chat))
        # chat = sigmoid(torch.abs(y_hid))
            # chat    = self.lin2(self.Rl1)
        return chat
#%% Triple layer:
# class GenNN(nn.Module):
    
#     def __init__(self,d_in,H1,H2,d_out):
#         super(GenNN, self).__init__()
#         self.lin1 = nn.Linear(d_in,H1)
#         self.lin2 = nn.Linear(H1,H2)
#         self.lin3 = nn.Linear(H2,d_out)
#     def forward(self,alpha):
#         # gel = nn.GELU()
#         self.a1 = self.lin1(alpha)
#         self.a2 = self.lin2(self.a1)
#         self.a3 = self.lin3(self.a2)
#         chat    = torch.exp(-self.a3**2)
#         return chat    

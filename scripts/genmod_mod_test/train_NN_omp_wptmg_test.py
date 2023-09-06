# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 20:23:57 2022

@author: jothi
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.nn as nn
import sys
#sys.path.append('/home/jothi/CoSaMP_genNN')
import genmod_mod_test.polynomial_chaos_utils as pcu
from ray.air.checkpoint import Checkpoint
from ray.air import session
from torch import linalg as LA
from ray import tune
import os
import genmod_mod_test.Gmodel_NN as gnn
#%% Train Model
def loss_crit(ghat,g,W_fc,f_x,ind_vt):
    P = ghat.size(dim=0)
    #weighted loss:
    # Wt = W_fc * ghat
    # Wt = f_x[ind_vt]
    if W_fc==1:
        Wt = torch.ones(P)
    else:
        Wt = W_fc * ghat
        #print("Wt:",Wt)
    # Wt_sqt = torch.sqrt(Wt)
    Lsqr_diff  = Wt*(ghat-g)**2
    L  = torch.sum(Lsqr_diff)    #weighted sum.
    L_uwt = torch.sum((ghat-g)**2)
    #other losses:
    # L  = torch.sum((ghat-g)**2)    
    # L  = torch.mean((ghat-g)**2)
    # L  = torch.sum(((ghat-g)**2))/(LA.norm(ghat))**2 #Relative error.
    # L  = torch.sum(((ghat-g)**2))/(torch.numel(ghat)) #Mean.
    # L  = torch.sum(((ghat-g)**2))/(LA.norm(ghat)*torch.numel(G_ini))
    # Loss_crt = nn.MSELoss(reduction='sum')
    return L,Wt,L_uwt
def train_theta(chat_omp,thet_up,thet_str1, Nt_ind, alph_in_tot,epochs,H,t_ind,v_ind,freq,W_fc,actlist,cnfg_tn,
        chkpnt_dir=None,data_dir=None):
   # print("sys path:",sys.path)
    
    cost = []
    zer_str = []
    cost_rel = []
    cost_abs = []
    cost_val = []
    cost_rel = []
    cost_val_rel = []
    cost_uwt = []
    cost_uwt_val = []
    # Loss_man = []
    # cer_str = []
    # ter_str = []
    total=0
    total_val_up = 1e8
    G_omp = chat_omp[t_ind]
    alph_in = alph_in_tot[t_ind,:]
    G_omp_val = chat_omp[v_ind]
    alph_in_val = alph_in_tot[v_ind,:]
    thet = thet_up
    thet_dict1 = thet.detach().numpy()
    P_in = alph_in.size(dim=0)
    d_in = alph_in.size(dim=1)
    #For weights:
    a = 0
    b = torch.amax(chat_omp)
    c = 0.5
    d = 30
    P_fl = chat_omp.size(dim=0)
    f_x = torch.zeros(P_fl)
    for i in range(P_fl):                 
        f_x[i] = c + (chat_omp[i]-a)*(d-c)/(b-a)        
    # d_in = 1
    # G_upd_dict = np.zeros(P_in)
    # thet = thet_up.view(-1,1)
    #GNNmod = gnn.GenNN(d_in,H,1)
    #optimizer = torch.optim.Adam(GNNmod.parameters(), lr=learning_rate)
    import pdb; pdb.set_trace()
    GNNmod = gnn.GenNN(cnfg_tn['Layers'])
    #GNNmod = gnn.GenNN([d_in,cnfg_tn['Hid'],1])
    optimizer = torch.optim.Adam(GNNmod.parameters(), lr=cnfg_tn['lr'])
    for epoch in range(epochs):
        total=0
        if epoch%1000==0: 
            print('epoch:',epoch)
        # print('thet_size:',thet.size())
        # print('thet_str_size:',thet_str.size())
        # for i in range(P):
        # if i==0:
        #     alph_in = torch.Tensor(mi_mat_omp1)
        # else:
        #     alph_in = multi_index_matrix
        if epoch==0:
            nn.utils.vector_to_parameters(thet, GNNmod.parameters())
            # thet_i = thet.detach().numpy() #it is gonna keep changing the parameters even if it is defined for epoch=0.
            G_ini = G_omp        
        G_NN = GNNmod(alph_in,actlist,Nt_ind).flatten()
        # G_NN_h = dmold.G_NN_nphrdcd(thet, alph_in,H)
        # print('G_NN',G_NN)
        # print('G_ver',G_ver)
        G_upd = G_NN.detach().numpy()
#        tp_20_G = np.argsort(G_upd)[::-1][:20]
#        if (tp_20_G==74).any():
#            df_thet_G74_tp20 = pd.DataFrame({'thet_up':thet_up_ep})
#            df_thet_G74_tp20.to_csv('/home/jothi/GenMod_omp/output/titan_ppr/plots/thet_G74_tp20_e{epoch}.csv')
#            df_G74_tp20 = pd.DataFrame({'G_upd':G_upd})                       
#            df_G74_tp20.to_csv('/home/jothi/GenMod_omp/output/titan_ppr/plots/G_upd_G74_tp20_e{epoch}.csv')       # G_upd_dict = np.vstack((G_upd_dict,G_upd))
        prm_dic1 = GNNmod.state_dict()
        # loss_man = torch.sum((G_NN-G_omp)**2)
        # loss_nn = nn.MSELoss(reduction='sum')
        # loss_iblt = loss_nn(G_omp,G_NN)
        loss,W_m,loss_uwt = loss_crit(G_omp,G_NN,W_fc,f_x,t_ind) #!!!!UNCOMMENT THE LOSS CONSISTENT WITH THE FUNCTION:
        loss.backward()
        # H=1, d=1: gradient check:        
        # dldthet_trch = torch.Tensor([GNNmod.lin1.weight.grad, GNNmod.lin1.bias.grad, GNNmod.lin2.weight.grad,
        #                               GNNmod.lin2.bias.grad])
        # thet_up_trch = thet_up - dldthet_trch * 0.001
        optimizer.step()
        optimizer.zero_grad()
        #cumulative loss 
        total=loss.item() 
        total_uwt = loss_uwt.item()
        # Validation Loss:
        G_NN_val = GNNmod(alph_in_val,actlist,Nt_ind).flatten()            
        loss_val, W_val,loss_uwt_val = loss_crit(G_omp_val,G_NN_val,W_fc,f_x,v_ind)
        total_val = loss_val.item()
        total_uwt_val = loss_uwt_val.item()
        # Zerr plot:
        thet_up_ep = nn.utils.parameters_to_vector(GNNmod.parameters())
        z_err = np.linalg.norm(thet_str1 - thet_up_ep.detach().numpy()) /np.linalg.norm(thet_str1)
        if epoch%freq==0:
           cost.append(total)       
           cost_abs.append(total)
           cost_val.append(total_val)
           cost_rel.append(total/la.norm(W_m*G_omp)**2)
           cost_val_rel.append(total_val/la.norm(W_val*G_omp_val)**2)
           cost_uwt.append(total_uwt)
           cost_uwt_val.append(total_uwt_val)
           zer_str.append(z_err)        
        if total_val < total_val_up:
           total_val_up = total_val #try to use torch.clone for copying.
           thet_bst = thet_up_ep
           ep_bst = epoch            
        else: 
           break
        checkpoint = Checkpoint.from_dict({"epoch": epoch,"thet":thet_up_ep.detach().numpy()})
        session.report({'loss_met':total_val,'train_loss':total,'z_err':z_err},checkpoint=checkpoint)
           #with tune.checkpoint_dir(epoch) as checkpoint_dir:
           #   path = os.path.join(checkpoint_dir, "checkpoint")
           #   #print("path:",path)
           #   torch.save((GNNmod.state_dict(), optimizer.state_dict()), path)
        # if epoch%1==0:
        # thet_dict1 = np.vstack((thet_dict1,thet_up_ep.detach().numpy()))   
        # debug=0
    thet_f = nn.utils.parameters_to_vector(GNNmod.parameters())
    print('epoch with minimum validation error:',ep_bst)
    #session.report({'cost':cost,'cost_val':cost_val,'thet_f':thet_f,'thet_bst':thet_bst})
    # plt.figure(1)
    # # c = next(color)
    # # plt.semilogy(np.linspace(0,epochs-1,epochs),cost,c=c)
    # plt.semilogy(np.linspace(0,epochs-1,epochs),cost,label='cost_itr = %s' % i)    
    # plt.legend()
    # print('cost',cost)
    #return {'cost':cost,'cost_val':cost_val,'thet_f':thet_f,'thet_bst':thet_bst}
def val_test_err(data_tst,mi_mat_t,c):
    y_data  = data_tst['y_data']
    u_data  = data_tst['u_data']    
    test_indices  = data_tst['test_ind']
    valid_indices  = data_tst['val_ind']
    Nv = data_tst['Nv']
    Lam_fnl = (np.nonzero(c)[0]).tolist() 
    c_lm = c[Lam_fnl]
    chc_Psi = data_tst['chc_poly'] 
    d = np.size(mi_mat_t,1)
 #   import pdb; pdb.set_trace()
    if chc_Psi == 'Legendre':
        Psi_test = pcu.make_Psi(y_data[test_indices, :d], mi_mat_t,chc_Psi)
        test_err = la.norm(
            Psi_test @ c - u_data[test_indices].T
        ) / la.norm(u_data[test_indices].T)
        # Validation error: is the error on the unseen data:
        Psi_valid = pcu.make_Psi(y_data[valid_indices[:Nv], :d], mi_mat_t,chc_Psi)
        valid_err = la.norm(
            Psi_valid @ c - u_data[valid_indices[:Nv]].T
        ) / la.norm(u_data[valid_indices[:Nv]].T)
    elif chc_Psi == 'Hermite':
        #Psi_test_fl = pcu.make_Psi(y_data[test_indices, :d], mi_mat_t,chc_Psi)
        Psi_test = pcu.make_Psi_drn(y_data[test_indices, :d], mi_mat_t,Lam_fnl,chc_Psi)
#        import pdb;pdb.set_trace()
#        test_err_fl = la.norm(
#            Psi_test_fl @ c - u_data[test_indices].T
#        ) / la.norm(u_data[test_indices].T)
        test_err = la.norm(
            Psi_test @ c_lm - u_data[test_indices].T
        ) / la.norm(u_data[test_indices].T)       
        # Validation error: is the error on the unseen data:
#        Psi_valid = pcu.make_Psi(y_data[valid_indices[:Nv], :d], mi_mat_t,chc_Psi)
        Psi_valid = pcu.make_Psi_drn(y_data[valid_indices[:Nv], :d], mi_mat_t,Lam_fnl,chc_Psi)
        valid_err = la.norm(
            Psi_valid @ c_lm - u_data[valid_indices[:Nv]].T
        ) / la.norm(u_data[valid_indices[:Nv]].T)
        
    return test_err, valid_err
def val_test_err_hmt(data_tst,mi_mat_t,c):
    y_data  = data_tst['y_data']
    u_data  = data_tst['u_data']    
    test_indices  = data_tst['test_ind']
    valid_indices  = data_tst['val_ind']
    Nv = data_tst['Nv']
    d = np.size(mi_mat_t,1)
    Psi_test = pcu.make_Psi_hermite(y_data[test_indices, :d], mi_mat_t)
    test_err = la.norm(
        Psi_test @ c - u_data[test_indices].T
    ) / la.norm(u_data[test_indices].T)
    # Validation error: is the error on the unseen data:
    Psi_valid = pcu.make_Psi_hermite(y_data[valid_indices[:Nv], :d], mi_mat_t)
    valid_err = la.norm(
        Psi_valid @ c - u_data[valid_indices[:Nv]].T
    ) / la.norm(u_data[valid_indices[:Nv]].T)
    return test_err, valid_err

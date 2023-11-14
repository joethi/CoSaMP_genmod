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
import math
import torch.nn as nn
import sys
#sys.path.append('/home/jothi/CoSaMP_genNN')
import genmod_mod_test.polynomial_chaos_utils as pcu
from ray.air.checkpoint import Checkpoint
from ray.air import session
from torch import linalg as LA
from ray import tune
import os
from  sklearn.model_selection import KFold
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
def kaiming_init_prms(model_obj):
    for name, prms in model_obj.named_parameters():
        #import pdb; pdb.set_trace()
        if name.endswith(".bias"):
            prms.data.fill_(0)
        #elif name.startswith("hidden.0") or name.startswith("hidden.1"):  # The first layer does not have ReLU applied on its input
        elif name.startswith("hidden.0"):  # The first layer does not have ReLU applied on its input
            prms.data.normal_(0, 1 / math.sqrt(prms.shape[1]))
        else:
            prms.data.normal_(0, math.sqrt(2) / math.sqrt(prms.shape[1]))

def train_theta(chat_omp,thet_up,thet_str1, Nt_ind, alph_in_tot,epochs,freq,W_fc,hidlist,actlist, Nlhid,TSIG,iter_fix,rnd_smp_dict,l_r, fr_hist, j_ind, chkpnt_dir=None,data_dir=None,p_d=0):
   # print("sys path:",sys.path)
    #import pdb; pdb.set_trace()
    cost = []
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
    total_trn_up = 1e8
    P_alg = alph_in_tot.size(dim=0)
    t_ind = rnd_smp_dict['trn_ind']    
    v_ind = rnd_smp_dict['val_ind']    
    G_omp = chat_omp[t_ind]
    alph_in = alph_in_tot[t_ind,:]
    G_omp_val = chat_omp[v_ind]
    alph_in_val = alph_in_tot[v_ind,:]
    #thet = torch.clone(thet_up)
    #thet_dict1 = thet.detach().numpy()
    thet_dict = {}    
    P_in = alph_in.size(dim=0)
    d_in = alph_in.size(dim=1)
    #For weights:
    a = 0
    b = torch.amax(chat_omp)
    c = 0.5
    de = 30
    P_fl = chat_omp.size(dim=0)
    f_x = torch.zeros(P_fl)
    for i in range(P_fl):                 
        f_x[i] = c + (chat_omp[i]-a)*(de-c)/(b-a)        

    GNNmod = gnn.GenNN([d_in] + hidlist +[1],p_d)
    #GNNmod = gnn.GenNN([d_in] + hidlist +[1])
    #GNNmod = gnn.GenNN([d_in,cnfg_tn['Hid'],1])
    optimizer = torch.optim.Adam(GNNmod.parameters(), lr=l_r)
    for epoch in range(epochs):
        if epoch==0:
            #print('thet_up',thet_up)
            GNNmod_dict = GNNmod.state_dict() 
            nprm_tnthet = sum(p_el.numel() for p_el in GNNmod_dict.values())            
            print("nprm_tnthet",nprm_tnthet)
            kaiming_init_prms(GNNmod)
            #nn.utils.vector_to_parameters(thet_up,GNNmod.parameters())
            G_NN_full = GNNmod(alph_in_tot,actlist,Nt_ind).flatten()      
            G_ini = G_NN_full.detach().numpy()
            #import pdb; pdb.set_trace()
        G_NN = GNNmod(alph_in,actlist,Nt_ind).flatten()
        G_upd = G_NN.detach().numpy()
        loss,W_m,loss_uwt = loss_crit(G_omp,G_NN,W_fc,f_x,t_ind) #!!!!UNCOMMENT THE LOSS CONSISTENT WITH THE FUNCTION:
        #===========================================================
        #cumulative loss 
        total=loss.item() 
        total_uwt = loss_uwt.item()
        #============calculate validation loss=====================
        G_NN_val = GNNmod(alph_in_val,actlist,Nt_ind).flatten()            
        loss_val, W_val,loss_uwt_val = loss_crit(G_omp_val,G_NN_val,W_fc,f_x,v_ind)
        total_val = loss_val.item()
        #===========================================================
        if epoch%freq==0: #and epoch<=iter_fix: #FIXME
           cost.append(total)       
           cost_val.append(total_val)
           #cost_rel.append(total/la.norm(W_m*G_omp)**2)
           cost_val_rel.append(total_val/la.norm(W_val*G_omp_val)**2)
        if (epoch+1)%fr_hist==0 or epoch<500: 
           thet_tmp  = nn.utils.parameters_to_vector(GNNmod.parameters())
           thet_dict[f'e{epoch}'] = thet_tmp.detach().numpy()
           print('epoch:',epoch,"total",total)
           print('epoch:',epoch,"total_val",total_val)
        if total_val < total_val_up:
           total_val_up = total_val #try to use torch.clone for copying.
           if epoch >0:  
               thet_bst = torch.clone(thet_up_ep)
           else:
               thet_bst = torch.clone(thet_up)
           ep_bst = epoch            
           total_trn_up = total #try to use torch.clone for copying.
        if epoch == epochs-1: 
            costval_min = min(cost_val) 
            costval_min_rel = min(cost_val_rel)
            #print('GNNmod dict:',GNNmod.state_dict())
            #checkpoint = Checkpoint.from_dict({"epoch": epoch,"thet_fx":GNNmod.state_dict(),"thet":thet_bst.detach().numpy(),"train_app":cost,"val_app":cost_val})
        #for debugging purposes:    
           # G_NN_full_f = GNNmod(alph_in_tot,actlist,Nt_ind).flatten()      
           # G_f = G_NN_full_f.detach().numpy()
            #session.report({'loss_met':costval_min+total_trn_up,'train_loss':total,'cost_val':total_val,'ep_best':ep_bst},checkpoint=checkpoint)
            print('costval_min',costval_min)
            print('epoch',epoch,'total_val_up',total_val_up)
            #print("epoch",epoch,"prm_ini_dict",prm_ini_dict)
            #print("epoch",epoch,"prm_fx_dict",prm_fx_dict)
            thet_f = nn.utils.parameters_to_vector(GNNmod.parameters())
        
        # calculate gradients and update parameters:
        #===========================================================
        if epoch < epochs-1:   
            loss.backward()
            optimizer.step()
            if (epoch+1)%10000==0 or epoch <10:    
                grad_dic = {x[0]:x[1].grad for x in GNNmod.named_parameters()}            
                torch.save(grad_dic,f'{chkpnt_dir}/plots/j={j_ind}/it={Nt_ind}/grad_dic_cpt_i{Nt_ind}_j{j_ind}_ep{epoch}.pt')
                #import pdb; pdb.set_trace()
            optimizer.zero_grad()
            thet_up_ep = nn.utils.parameters_to_vector(GNNmod.parameters())
        #===========================================================
        #import pdb; pdb.set_trace()
    print('epoch with minimum validation error:',ep_bst)
    #import pdb; pdb.set_trace()
    #return {'cost':cost,'cost_val':cost_val,'thet_f':thet_f,'thet_bst':thet_bst,'ep_bst':ep_bst,'costval_min':costval_min,"G_f":G_f}, thet_dict
    return {'cost':cost,'cost_val':cost_val,'thet_f':thet_f,'thet_bst':thet_bst,'ep_bst':ep_bst,'costval_min':costval_min,'costval_min_rel':costval_min_rel}, thet_dict
def get_best_result_from_kfoldcv(results_kcv):
    nfold = len(results_kcv)
    min_vlss = np.zeros(nfold)
    for i in range(nfold):
        min_vlss[i] = results_kcv[i]['costval_min_rel']        
    bst_ind = np.argmin(min_vlss)        
    #import pdb; pdb.set_trace()    
    return bst_ind,min_vlss       
def kfoldcv_manual_stratified_split(cini_nz,cini_z,nfld_trn=5,rnd_st_cvtrn=42):
    cini_nz_ln = np.size(cini_nz)    
    cini_z_ln = np.size(cini_z)                    
    P = cini_nz_ln + cini_z_ln     
    kf = KFold(n_splits=nfld_trn,shuffle=True,random_state=rnd_st_cvtrn)    
    tind_z_allfld = []   
    for i_fld, (tind_z_fld, vind_z_fld) in enumerate(kf.split(cini_z)):
       tind_z_allfld.append((cini_z[tind_z_fld], cini_z[vind_z_fld]))       
       #import pdb; pdb.set_trace()    
    # handling remaining zero indices:
    #z_splt_int = int(cini_z_ln/5)
    #rem_z = cini_z_ln - z_splt_int * 5     
    # remaining non-zero indices:
    tind_nz_allfld = []   
    for i1_fld, (tind_nz_fld, vind_nz_fld) in enumerate(kf.split(cini_nz)):
       tind_nz_allfld.append((cini_nz[tind_nz_fld], cini_nz[vind_nz_fld]))       
       print("valid + train",np.union1d(tind_nz_allfld[i1_fld][0],tind_nz_allfld[i1_fld][1]))     
    tind_full = []    
    #import pdb; pdb.set_trace()    
    for i_fld_f in range(nfld_trn):
       tind_tmp = np.concatenate((tind_z_allfld[i_fld_f][0], tind_nz_allfld[i_fld_f][0]))          
       vind_tmp = np.concatenate((tind_z_allfld[i_fld_f][1], tind_nz_allfld[i_fld_f][1]))          
       tind_full.append((tind_tmp,vind_tmp))
    #debugging/verification:
    #for i in range(5):
    #  tot = np.concatenate((tind_full[i][0],tind_full[i][1]))  
    #  print("train_ind fraction",np.round(np.size(tind_full[i][0]) / P,4))  
    #  print("val_ind fraction",np.round(np.size(tind_full[i][1]) / P,4))  
    #  print(f"i={i},total - split (train+val)", np.setdiff1d(np.arange(0,P),tot))  

    #import pdb; pdb.set_trace()    
    #nz_splt_int = int(cini_nz_ln/5)   
    #rem_nz = cini_nz_ln - nz_splt_int * 5     
    #for i in range(nfld_trn):    
    #    tind_z_allfld[i][]    
        
    return tind_full    
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

# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 16:30:20 2022

@author: jothi
"""
# coding: utf-8

# # Orthogonal Matching Pursuit (OMP)
# reference: Sergios' Machine Learning book, Chapt.10
import cProfile
import scipy.io as sio
import seaborn as sns
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import csv
import random
import torch
import torch.nn as nn
import sys
import time
import pickle
import statistics as sts
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import sklearn.linear_model as lm
from itertools import combinations
from functools import partial
from ray.tune import CLIReporter
from ray.air.checkpoint import Checkpoint
#from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import AsyncHyperBandScheduler 
from ray import tune, air
from  sklearn.model_selection import StratifiedKFold, KFold
import multiprocessing
import ray
import argparse
import os
# from numpy import log10, sign, abs
np.random.seed(1)
# torch.manual_seed(0)
sys.path.append('/home/jothi/CoSaMP_genNN/no_ray')
#import pdb;pdb.set_trace()
sys.path.append('/home/jothi/CoSaMP_genNN/no_ray/scripts/GenMod-org-Hmt')
#sys.path.append('/home/jothi/GenMod_omp/scikit-learn-main/sklearn/linear_model')
# sys.path.append('C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod_omp')
# out_dir_ini = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod_omp/output/duff_osc_ppr'
# out_dir_ini = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod_omp/output/wing_wght'
# out_dir_ini = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod_omp/output/1DElliptic_ppr'
#out_dir_ini = f'../output/titan_ppr/results'
# Redirect stdout to a file
#sys.stdout = open(f'{out_dir_ini}/plots/log_printed.txt', 'w')
import genmod.run_optimizations_rsdl as ro
import genmod_mod_test.polynomial_chaos_utils as pcu
import genmod_mod_test.Gmodel_NN as gnn
import genmod_mod_test.train_NN_omp_wptmg_test as tnn
import genmod_mod_test.omp_utils as omu
import genmod_mod_test.test_coeffs_val_er_utils as tcu
import warnings
#import omp as omp1
#from genmod_mod import Gmodel_NN as gnn_test
#import _omp as omp_scd
#import pdb;pdb.set_trace()

#=================================================================================
#=================================================================================
def mo_main_utils_function_prll(data_all,out_dir_ini,opt_params,nn_prms_dict,indices0,args,eps_u,W_fac,eps_abs,j):
    p = opt_params['ph']; p_0 = opt_params['p0'];d = opt_params['d'];epochs = opt_params['epochs'];
    learning_rate = opt_params['lr']; S_omp = opt_params['Sh'];S_omp0 = opt_params['S0'];
    tot_itr = opt_params['N_t'];freq = opt_params['fr'];#W_fac = opt_params['W_fac'] #TODO] 
    z_n = opt_params['z_n']; top_i1 = opt_params['Tp_i1'];top_i0 = opt_params['Tp_i0'];N = opt_params['N']  
    Nv = opt_params['Nv'];Nrep = opt_params['Nrep'];Nc_rp = opt_params['Nc_rp'];S_chs = opt_params['S_chs']  
    chc_Psi = opt_params['chc_poly']; seed_ind = opt_params['sd_ind']; seed_thtini = opt_params['sd_thtini'] 
    seed_ceff = opt_params['sd_ceff'];Nrp_vl = opt_params['Nrp_vl'];sd_thtini_2nd = opt_params['sd_thtini_2nd']   
    it_fix = opt_params['iter_fix']; num_trial = opt_params['ntrial'];chc_omp_slv = opt_params['chc_omp_slv'] 
    Nlhid = opt_params['Nlhid']; sprsty = opt_params['sprsty']; chc_eps = opt_params['chc_eps']
    y_data = data_all['y_data']; u_data = data_all['u_data']; mi_mat = data_all['mi_mat']; P = opt_params['P']     
    avtnlst = nn_prms_dict['avtnlst']; hid_layers = nn_prms_dict['hid_layers']; tune_sg = nn_prms_dict['tune_sg']  
    print('j',j,'W_fac',W_fac,'type of W_fac',type(W_fac))
    #import pdb; pdb.set_trace()    
    print(f'=============#replication={j}============')
    ecmn_ind = np.zeros(tot_itr)
    os.makedirs(f'{out_dir_ini}/plots/j={j}',exist_ok=True)
    optim_indices = indices0.iloc[j].to_numpy()
    valid_indices = np.setdiff1d(range(np.size(u_data)), optim_indices)
    trains = [name for name in indices0.columns if name.startswith("optim")]
    test_indices = indices0.loc[j][trains].to_numpy()
    nfld_ls = args.Nfld_ls    
    nfld_trn = args.Nfld_trn  
    rnd_st_cvls = args.rnd_st_cvls
#    import pdb; pdb.set_trace()
    data_tst = {'y_data':y_data,'u_data':u_data,'val_ind':valid_indices,'test_ind':test_indices,'opt_ind':optim_indices,'Nv':Nv,
            'chc_poly':chc_Psi,'chc_omp':chc_omp_slv} 
#======================================================================================
#======================================================================================
#======================================================================================
    #import pdb; pdb.set_trace()
    mo_time_strt = time.time()
    if args.debug_alg==1:
        #use this to use user-specifc c_hat values.
        c_ini, S_omp0, train_err_p0, valid_err_p0,P_omp,mi_mat_omp, Psi_omp = omu.omp_utils_order_ph_dummy(out_dir_ini,args.cht_ini_fl,d,p_0,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_omp0,j)
    elif args.use_gmd==1:
        print("inside genmod loop")
        opt_lst = [*[f'optim.{t_in}' for t_in in range(int(N*4/5))],*[f'valid.{v_in}' for v_in in range(int(N/5))]]
        opt_params_gmd = {'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'stepSize': 0.001,
                      'maxIter': 100000, 'objecTol': 1e-6, 'ALIter': 10,
                      'resultCheckFreq': 10, 'updateLambda': True,
                      'switchSigns': False, 'useLCurve': False, 'showLCurve': False,'Nvlrp': 1}
        df_opt_params_gmd = pd.DataFrame(opt_params_gmd,index=[0])
        df_opt_params_gmd.to_csv(f'{out_dir_ini}/params_genmod_org_adam_N={N}.csv')
        indices_gmd = indices0.copy()
        indices_gmd = indices_gmd.set_axis(opt_lst,axis=1)
        
        #import pdb;pdb.set_trace()
        c_ini,Psi_omp =  ro.run_genmod(j_rng[0], j_rng[0],'1dellps_gmdorg_n=' + str(N), d, p_0, data_all,indices_gmd,
              f'{out_dir_ini}/plots', N, Nv, chc_Psi,mi_mat_p0,2*d+1, lasso_eps=1e-10,
              lasso_iter=1e5, lasso_tol=1e-4, lasso_n_alphas=100,
              opt_params=opt_params_gmd)
        #df_cgmd = pd.read_csv('../output/titan_ppr/results/csaug13/d=21/p=3/ref_dbg/gmd_org/1dellps_gmdorg_n=100_genmod_kmin=0_1.csv')
        #c_ini = df_cgmd['Coefficients'].to_numpy().flatten()
        mi_mat_omp = np.copy(mi_mat_p0)
        P_omp = np.size(mi_mat_omp,0)
        #S_omp0 = np.nonzero(c_ini) 
        train_err_p0, valid_err_p0 = tnn.val_test_err(data_tst,mi_mat_omp,c_ini)
    else:
        c_ini, S_omp0, train_err_p0, valid_err_p0,P_omp,mi_mat_omp, Psi_omp = omu.omp_utils_order_ph(out_dir_ini,d,p_0,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_omp0,j)
    if args.omp_only==1:
        print('OMP calculations were done---breaking as requested!')
        return
    #import pdb;pdb.set_trace()
#   
##Calculate S for 0 and h:
#    if chc_eps == 'c':
#        S_refh = np.size(np.nonzero(c_ref[:P])[0])
#        S_ref0 = np.size(np.nonzero(c_ref[:P_omp])[0])
#    # c_ini = c_gen[:P_omp]
#=============================================================================
# Find omp coefficients for the higher order omp:
#=============================================================================
    omph_time_strt = time.time()
    c_omph, S_omph, test_omp_ph, valid_omp_ph,P_omph,mi_mat_omph, Psi_omph= omu.omp_utils_order_ph(out_dir_ini,d,p,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_omp,j)
#=============================================================================
    omph_time_end = time.time()
    print("omph time:",omph_time_end-omph_time_strt)
    #import pdb;pdb.set_trace()
    #print()    
    # #p=3
    print('S_omph:',S_omph)
    #print(f'Training Error for c_omph: {test_omp_ph}')
    ## Validation error: is the error on the unseen data:
    #print(f'Validation Error for c_omph: {valid_omp_ph}')
    #import pdb;pdb.set_trace()
# Least squares:
    # eps_u_tomp = la.norm(Psi[test_indices[:N_tep],:] @ c_omph - u_data[test_indices[:N_tep]])/la.norm(u_data[test_indices[:N_tep]])
    if chc_eps =='c':
        eps_c_omp = la.norm(c_omph - c_ref)    
        eps_c_omp_abs.append(la.norm(c_omph - c_ref))    
        epsc_omph.append(eps_c_omp)
    df_epscomp = pd.DataFrame({'epsu_omph':valid_omp_ph,'epsu_omph_t':test_omp_ph},index=[0])
    df_epscomp.to_csv(f'{out_dir_ini}/plots/epsuomph_tst_1dellps_n={N}_genmod_S={S_omph}_j{j}.csv',index=False)
    #import pdb; pdb.set_trace()
    #=============================================================================
#    import pdb; pdb.set_trace()
    plt.figure(110)
    max_nnzr = np.size(np.nonzero(c_ini))
    plt.plot(np.nonzero(c_ini),np.nonzero(c_ini),'*r')
    plt.xlabel('Index of PCE coefficients')
    plt.ylabel('Active sets')  
    #%% initial parameters:
    multi_ind_mtrx = torch.Tensor(mi_mat)
#    print('GNNmod:',GNNmod)
    np.random.seed(seed_thtini)
    thet_str = np.random.rand(z_n)
    thet_upd = torch.Tensor(thet_str)
    # thet_upd = torch.Tensor(thet_str+0*np.random.randn(z_n)) #0.1*np.random.randn(z_n) 
    # color = iter(cm.rainbow(np.linspace(0, 1, tot_itr)))
    opt_params = {'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'stepSize': 0.001,
                  'maxIter': 100000, 'objecTol': 1e-6, 'ALIter': 1,
                  'resultCheckFreq': 10, 'updateLambda': True,
                  'switchSigns': False, 'useLCurve': False, 'showLCurve': False,'batch_size': None}
    opt_params['nAlphas'] = 100
    opt_params['tolLasso'] = 1e-4
    opt_params['iterLasso'] = 1e5
    opt_params['epsLasso'] = 1e-10
    #% Save parameters:
    opt_params = {'ph':p,'p0':p_0,'d':d,'epochs':epochs,'lr':learning_rate,'S':S_omp,'S0':S_omp0,
                  'tot_it':tot_itr,'fr':freq,'W_fac':f'{W_fac}'}
#    import pdb;pdb.set_trace() 
    df_params = pd.DataFrame(opt_params,index=[0])
    df_params.to_csv(f'{out_dir_ini}/plots/j={j}/params_genmod_omp_N={N}.csv')
    print(df_params)
    df_thet_str = pd.DataFrame({'thet_str':thet_str})
    df_thet_str.to_csv(f'{out_dir_ini}/plots/j={j}/thet_str_genmod_omp_N={N}.csv')
    #%%
    #cost_tot = np.zeros(int(epochs/freq))
    cost_rel_tot = np.zeros(int(epochs/freq))
    z_err_tot = np.zeros(int(epochs/freq))
    
    thet_dict = np.zeros_like(thet_str)
    Gmod_dict = np.zeros((P,1))
    test_err_ls = []
    valid_err_ls = []
    c_ini_full = np.zeros(P)
    c_omp_fl = np.zeros(P)
    c_omp_fl[:P_omp] = c_ini
    # eps_c.append(la.norm(c_omp_fl - c_ref)/la.norm(c_ref))
    if chc_eps == 'c':
        eps_c[0] = la.norm(c_omp_fl - c_ref)
        # eps_c[0] = la.norm(c_omp_fl - c_ref)
        eps_abs.append(la.norm(c_omp_fl - c_ref))
    c_omp_bst = np.zeros(P)   
    eps_ctmp = np.zeros(Nc_rp)
    epsu_tmp = np.zeros(Nc_rp)
    #%% Least squares
    #%% Define the training function:
    #initialization:
    #nn.utils.vector_to_parameters(thet_upd, GNNmod.parameters())
    #print("GNNmod parameters:",GNNmod.parameters())
    #Train on a few points:
    mi_mat_in = np.copy(mi_mat_omp)
    c_hat = np.copy(c_ini)
    cr_mxind = (np.argsort(np.abs(c_hat))[::-1])[:top_i0]
    # mi_mat_in = mi_mat_omp[S_fnl,:]
    # c_hat = c_ini[S_fnl]
#    test_err, valid_err = tnn.val_test_err(data_tst,mi_mat_omp,c_ini)
    eps_u[0] = valid_err_p0
    
    #%% Start training:
    ls_vlit_min = np.zeros((Nrp_vl,tot_itr))
    #import pdb; pdb.set_trace()    
    #initalize i:    
    i = 0
    rsdl_nrm_it = []
    while i < tot_itr:
        print(f'=============total iteration index={i}============')
        os.makedirs(f'{out_dir_ini}/plots/j={j}/it={i}',exist_ok=True)
        # print('G_mod:',G_mod) 
        for trc in range(Nc_rp):
            if i>0:
                mi_mat_in = np.copy(mi_mat)   
                #uncomment to run the code with the corresponding validation coefficients used for the previous iteration:
                # c_hat = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/comp_fl_1dellps_n={N}_genmod_S={S_omp}_{i-1}_j{j}_c{trc}.csv').to_numpy().flatten()   
                c_hat = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/comp_fl_1dellps_n={N}_genmod_S={S_omp}_{i-1}_j{j}_c{int(ecmn_ind[i-1])}.csv').to_numpy().flatten()
                # NOTe: Here you won't be using more than Nc_rp=1.
                #c_hat = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/comp_rs_1dellps_n={N}_genmod_S={S_omp}_{i-1}_j{j}_c{trc}.csv').to_numpy().flatten()
                cr_mxind = (np.argsort(np.abs(c_hat))[::-1])[:top_i1]

            #import pdb; pdb.set_trace()
            P_alg = np.size(c_hat)
            cini_nz = np.nonzero(c_hat)[0]
            cini_nz_ln = np.size(cini_nz) 
            cini_z = np.setdiff1d(range(P_alg),cini_nz)
            cini_z_ln = np.size(cini_z)
            print('cini_nz',cini_nz)
            #=============================================================
            #code for debugging the activation functions, number of hidden layers: 
            #=============================================================
            #if args.debug_alg==2:
            if args.dbg_rdtvind==1:
                ##df_trn_ind = pd.read_csv('/home/jothi/CoSaMP_genNN/output/titan_ppr/results/csaug13/d=21/p=3/ref_dbg/plots_abs/trn_indices_alph_omp_N=100_0_c0.csv')
                ##t_indnz_sbmx_dbg = [3,4,16,196]
                #df_trn_ind = pd.read_csv('/home/jothi/CoSaMP_genNN/output/titan_ppr/results/d78_ppr/ref_dbg/trn_indices_alph_omp_N=80_0_c0.csv')
                #trn_ind_dbg_fl = df_trn_ind['trn_ind_nw'].to_numpy() 
                #t_indnz_sbmx_dbg = [0,2,9,60]
                #trn_ind_dbg_1 = np.setdiff1d(trn_ind_dbg_fl,cr_mxind)
                #trn_indz_dbg = np.setdiff1d(trn_ind_dbg_1,t_indnz_sbmx_dbg) 
                #config_tune['tind_nz'] = tune.sample_from(lambda _:t_indnz_sbmx_dbg)
                #config_tune['tind_z'] =  tune.sample_from(lambda _:trn_indz_dbg) 
                debug_num = 0
                #config_tune['tind_nz'] = tune.sample_from(lambda _:cini_nz)
                #config_tune['tind_z'] =  tune.sample_from(lambda _:cini_z) 
            else:
                debug_num = 0
            #print('config_tune',config_tune)
            #=
           # import pdb;pdb.set_trace()
            #FIXME: the training stops once the validation error starts increasing: sometimes, you might want to use more epochs. 
            if args.dbg_it2==1:
                if i>0:
                    debug_num = 0
                    #hid_layers = [tune.randint(7,8) for __ in range(Nlhid)]    
                    #avtnlst =[tune.choice([nn.ReLU()]),tune.choice(['None'])]
            #function doesn't involve ray:
            #=============================================================
            #Manual kfold split:
            #=============================================================

            results_kcv = []
            thet_dict_full = []
            tind_allfld_tpl = tnn.kfoldcv_manual_stratified_split(cini_nz,cini_z) 
            #import pdb;pdb.set_trace() 
            for i_fltnz, (trn_ind_fl_tr,tst_ind_fl_tr) in enumerate(tind_allfld_tpl):
                print("============================================================")
                print(f"=============fold-{i_fltnz}===============")
                print("============================================================")
                print("common indices between train and c_hat (nz)", np.intersect1d(trn_ind_fl_tr,cini_nz))
                print("common indices between test and c_hat (nz)", np.intersect1d(tst_ind_fl_tr,cini_nz))
                #import pdb;pdb.set_trace() 
                rnd_smp_dict = {'trn_ind':trn_ind_fl_tr,'val_ind':tst_ind_fl_tr}
                res_dict,thet_hist  = tnn.train_theta(torch.Tensor(np.abs(c_hat.flatten())),thet_upd,thet_str,i, 
                                      torch.Tensor(mi_mat_in),epochs,freq,W_fac[i],hid_layers,avtnlst,
                                      Nlhid,tune_sg,it_fix,rnd_smp_dict,learning_rate,args.fr_hist,
                                      j,chkpnt_dir=out_dir_ini)
                results_kcv.append(res_dict)      
                thet_dict_full.append(thet_hist) 
                #thet_upd = torch.Tensor(np.random.rand(z_n))    
                thet_upd = torch.Tensor(thet_str)
            #=============================================================
            # for intentional overfitting:
            #=============================================================
            #results_kcv = []
            #thet_dict_full = []
            ##import pdb;pdb.set_trace() 
            #    
            #trn_ind_fl_tr = np.arange(0,P_alg)
            #tst_ind_fl_tr = np.array([]) 
            #tind_allfld_tpl = [(trn_ind_fl_tr,tst_ind_fl_tr)]
            #rnd_smp_dict = {'trn_ind':trn_ind_fl_tr,'val_ind':tst_ind_fl_tr}
            #res_dict,thet_hist  = tnn.train_theta(torch.Tensor(np.abs(c_hat.flatten())),thet_upd,thet_str,i, 
            #                      torch.Tensor(mi_mat_in),epochs,freq,W_fac[i],hid_layers,avtnlst,
            #                      Nlhid,tune_sg,it_fix,rnd_smp_dict,learning_rate,args.fr_hist,
            #                      j,chkpnt_dir=out_dir_ini)
            #results_kcv.append(res_dict)      
            #thet_dict_full.append(thet_hist) 
            ##thet_upd = torch.Tensor(np.random.rand(z_n))    
            #thet_upd = torch.Tensor(thet_str)
            #=============================================================
                #if i==0:    
                #    #to avoid 
                #    thet_upd = torch.Tensor(thet_str)
                #else:
                #    np.random.seed(i+sd_thtini_2nd)
                #    thet_upd = torch.Tensor(np.random.rand(z_n))    
                #import pdb;pdb.set_trace() 
            #import pdb;pdb.set_trace() 
            #=============================================================
            # StratifiedKFold only works with categorical data, so you need to create categories
            # for the nonzero elements
            #=============================================================
            #categories = [1] *cini_nz_ln + [0] *cini_z_ln 
            ## Combine the indices of the nonzero and zero elements and their respective categories
            #all_indices = np.concatenate((cini_nz,cini_z))
            #all_categories = np.array(categories)
            #kf_trn = StratifiedKFold(n_splits=nfld_trn,shuffle=True,random_state=args.rnd_st_cvtrn)
            #results_kcv = []
            #for i_fltnz, (trn_ind_fltnz,tst_ind_fltnz) in enumerate(kf_trn.split(all_indices,all_categories)):
            #    print("============================================================")
            #    print(f"=============fold-{i_fltnz}===============")
            #    print("============================================================")
            #    #print("train set:",trn_ind_fltnz)                
            #    #print("val set:",tst_ind_fltnz)                
            #    print('c_hat (train):',c_hat[trn_ind_fltnz])
            #    print("common indices between train and c_hat (nz)", np.intersect1d(trn_ind_fltnz,cini_nz))
            #    print("common indices between test and c_hat (nz)", np.intersect1d(tst_ind_fltnz,cini_nz))
            #    rnd_smp_dict = {'trn_ind':trn_ind_fltnz,'val_ind':tst_ind_fltnz}
            #    res_dict  = tnn.train_theta(torch.Tensor(np.abs(c_hat.flatten())),thet_upd,thet_str,i, 
            #                          torch.Tensor(mi_mat_in),epochs,freq,W_fac[i],hid_layers,avtnlst,
            #                          Nlhid,tune_sg,it_fix,rnd_smp_dict,learning_rate)
            #    results_kcv.append(res_dict)      
            #    import pdb;pdb.set_trace() 
            best_ind_kcv, mn_vl_ls_lst = tnn.get_best_result_from_kfoldcv(results_kcv)  
            best_result_dict = results_kcv[best_ind_kcv] 
            #dump full fold results into pickle:
            best_res_full_dump = {'res_kcv':results_kcv,'full_thet':thet_dict_full,'tvind':tind_allfld_tpl} 
            with open(f'{out_dir_ini}/plots/j={j}/it={i}/full_results.pickle','wb') as bprms_pickl:
                pickle.dump(best_res_full_dump,bprms_pickl)
                
            #print(":after partfunc:")
            ##tuner = tune.Tuner(part_fnc,run_config=air.RunConfig(storage_path=f'{out_dir_ini}') ,tune_config=tune.TuneConfig(metric='loss_met', mode='min',scheduler=scheduler,num_samples=num_trial), param_space=config_tune)
            ##import pdb;pdb.set_trace()
            #print(":after results:")
            #result = tuner.fit() 
            #result_df = result.get_dataframe() 
            #best_result = result.get_best_result()
            #best_config = best_result.config
            #with open(f'{out_dir_ini}/plots/j={j}/it={i}/best_config.pickle','wb') as bprms_pickl:
            #    pickle.dump(best_config,bprms_pickl)
            #print("Best hyperparameters found were: ",best_config)
            #GNNmod = gnn.GenNN([d] + [best_config[] for lyr in range(Nlhid)] +[1])
            #GNNmod = gnn.GenNN([d] + hid_layers +[1])
            GNNmod = gnn.GenNN([d] + hid_layers +[1],args.p_d)
            #torch.save(GNNmod.state_dict(),f'{out_dir_ini}/plots/j={j}/it={i}/modelprms_final_dict_i{i}_j{j}.pt')
            #import pdb; pdb.set_trace()
            #best_result_df = best_result.metrics_dataframe
            best_epoch = best_result_dict['ep_bst'] 
            #cost = best_result_df['train_loss'].to_numpy().flatten() 
            #retrieve the best model:
            #best_checkpoint = best_result.checkpoint  
            #best_chckpnt_dict = best_checkpoint.to_dict() 
            thet_bst = best_result_dict['thet_bst']      
            cost = best_result_dict['cost']      
            cost_val = best_result_dict['cost_val']      
            #cost_val = best_chckpnt_dict.get("val_app")      
            torch.save(best_result_dict,f'{out_dir_ini}/plots/j={j}/it={i}/model_best_cpt_i{i}_j{j}.pt')
            #predict the third order coefficients:
            #Gmod_bst = GNNmod(torch.Tensor(mi_mat),[best_config.get(f'a{lyr1}') for lyr1 in range(Nlhid)],i).detach().numpy().flatten()
            nn.utils.vector_to_parameters(thet_bst, GNNmod.parameters())
            Gmod_bst = GNNmod(torch.Tensor(mi_mat),avtnlst,i).detach().numpy().flatten()
            #import pdb; pdb.set_trace()
            #TODO : start correcting the code from here
            # I want you to use checkpoints to save the nn parameters.    
            #check if the nn parameters flattening becomes problematic 
            #retrives initial checkpoint:
            #bcpt_path = best_checkpoint.path
            #print('bcpt_path',bcpt_path)
            #index_cpt = bcpt_path.find("checkpoint_")
            #index_uscr = index_cpt+len("checkpoint_")
            #add_strng_pth =  str(0).zfill(len(bcpt_path[index_uscr:]))
            #ini_path = bcpt_path[:index_uscr] + add_strng_pth
            #inl_cptdir  = Checkpoint.from_directory(ini_path)
            #inl_cpt_dict = inl_cptdir.to_dict()
            #torch.save(inl_cpt_dict,f'{out_dir_ini}/plots/j={j}/it={i}/model_ini_cpt_i{i}_j{j}.pt')
            ##import pdb; pdb.set_trace()
            ##retrives final checkpoint:
            #add_strng_pth_fnl =  str(1).zfill(len(bcpt_path[index_uscr:]))
            #final_path = bcpt_path[:index_uscr] +add_strng_pth_fnl
            #final_cptdir  = Checkpoint.from_directory(final_path)
            #final_cpt_dict = final_cptdir.to_dict()
            #torch.save(final_cpt_dict,f'{out_dir_ini}/plots/j={j}/it={i}/model_fnl_cpt_i{i}_j{j}.pt')
            #import pdb; pdb.set_trace()
            #FIXME:seeding for initializing theta in the next iteration
            #thet_upd = torch.Tensor(thet_bst)
# Write temp    orary variables:
            ## Least squares step at low validation error:          
            c_omp_bst = np.zeros(P)
            c_omp_sel = np.zeros(P)
#====================NOTe HERE thet_upd was there in the place of thet_bst=========            
            #nn.utils.vector_to_parameters(torch.Tensor(thet_bst), GNNmod.parameters())
            #torch.save(GNNmod,f'{out_dir_ini}/plots/j={j}/it={i}/model_final_i{i}_j{j}.pt')
            #torch.save(GNNmod.state_dict(),f'{out_dir_ini}/plots/j={j}/it={i}/modelprms_final_dict_i{i}_j{j}.pt')
            #Gmod_bst = GNNmod(torch.Tensor(mi_mat),[best_config.get(f'a{lyr1}') for lyr1 in range(Nlhid)],i).detach().numpy().flatten()
#=================For randomly initializing \theta (I think it is redundant)=================================            
#            nn.utils.vector_to_parameters(thet_upd, GNNmod.parameters())           
#Test the idea of picking many basis functions and keeping only the most important:
           # Psi_test = pcu.make_Psi(y_data[optim_indices,:d],mi_mat,chc_Psi)
#                import pdb; pdb.set_trace()
#                Lambda_bst = (np.argsort(Gmod_bst)[::-1])[:S_omp]
#                Psi_test = pcu.make_Psi(y_data[optim_indices,:d],mi_mat,chc_Psi)
#%% approximate residual signal:
            if args.add_tpso_res==0: 
                if i==0:
                    Lambda_sel_tmp = (np.argsort(Gmod_bst)[::-1])[:S_chs]               
                    Lambda_sel = Lambda_sel_tmp
                else:
                    c_res = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/comp_rs_1dellps_n={N}_genmod_S={S_omp}_{i-1}_j{j}_c{trc}.csv').to_numpy().flatten()
                    crs_ind_tp = np.argsort(np.abs(c_res))[::-1][:args.sel_res]
                    G_bst_h = Gmod_bst[P_omp:]
                    tpgh_ind = np.argsort(G_bst_h)[::-1][:S_chs] 
                    Lambda_sel_tmp = tpgh_ind + P_omp    
                    Lam_pr_bst  = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/Lam_bst_1dellps_n={N}_genmod_S={S_omp}_{i-1}_j{j}_c{trc}.csv')['Lam_bst'].to_numpy().flatten()
                    Lam_comn1 = np.intersect1d(Lambda_sel_tmp,Lam_pr_bst)
                    print('Lam_comn1',Lam_comn1)
                    Lam_comn2 = np.intersect1d(crs_ind_tp,Lam_pr_bst)
                    print('Lam_comn2',Lam_comn2)
                    Lam_comn = np.union1d(Lam_comn1,Lam_comn2)  
                    print('Lam_comn',Lam_comn)
                    S_comn = Lam_comn.size
                    #S_csit = S_omp+S_chs-S_comn
                    print('tpgh_ind',tpgh_ind)
                    print('G_bst_h',G_bst_h)
                    print('crs_ind_tp',crs_ind_tp)
                    print("Lambda_sel_tmp",Lambda_sel_tmp)
                    print("Lambda_pr_bst",Lam_pr_bst)
                    print("Lam_comn",Lam_comn)

                    if np.size(Lam_comn1) > 0 and np.size(Lam_comn2)==0:
                        Lambda_sel_tmp_g = Lambda_sel_tmp[np.in1d(Lambda_sel_tmp, Lam_comn1, invert=True)][:sprsty-args.sel_res]       
                        Lambda_sel_tmp = np.union1d(Lambda_sel_tmp_g,crs_ind_tp)
                        
                    elif np.size(Lam_comn1) == 0 and np.size(Lam_comn2) > 0:
                        Lc2_sz = np.size(Lam_comn2)   
                        add_res_unq = np.setdiff1d(crs_ind_tp,Lam_comn2)
                                #import pdb; pdb.set_trace()
                        Lambda_sel_tmp = np.union1d(Lambda_sel_tmp[:sprsty-args.sel_res+Lc2_sz],add_res_unq) 
                    elif S_comn==0: 
                        Lambda_sel_tmp = np.union1d(Lambda_sel_tmp[:sprsty-args.sel_res],crs_ind_tp) 
                    print("Lambda_sel_tmp",Lambda_sel_tmp)
                    Lambda_sel = np.union1d(Lambda_sel_tmp,Lam_pr_bst)
                    print("Lambda_sel",Lambda_sel)
                    #import pdb; pdb.set_trace()
                    #previous approach:    
                    #Lambda_sel_tmp = (np.argsort(Gmod_bst)[::-1])[:S_chs]
                    #Lam_pr_bst  = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/Lam_bst_1dellps_n={N}_genmod_S={S_omp}_{i-1}_j{j}_c{trc}.csv')['Lam_bst'].to_numpy().flatten()
                    #Lam_comn = np.intersect1d(Lambda_sel_tmp,Lam_pr_bst)
                    #S_comn = Lam_comn.size
                    #S_csit = S_omp+S_chs-S_comn
                    #print("Lambda_sel_tmp",Lambda_sel_tmp)
                    #print("Lambda_pr_bst",Lam_pr_bst)
                    #print("Lam_comn",Lam_comn)

                    #if S_csit > S_chs:
                    #    Lambda_sel_tmp = Lambda_sel_tmp[np.in1d(Lambda_sel_tmp, Lam_comn, invert=True)][:sprsty]       
                    #    #import pdb; pdb.set_trace()
                    #elif S_comn==0: 
                    #    Lambda_sel_tmp = Lambda_sel_tmp[:sprsty]
                    #print("Lambda_sel_tmp",Lambda_sel_tmp)
                    #Lambda_sel = np.union1d(Lambda_sel_tmp,Lam_pr_bst)
                    #print("Lambda_sel",Lambda_sel)
            elif args.add_tpso_res==2: #very vanilla approach
                if i==0:
                    Lambda_sel_tmp = (np.argsort(Gmod_bst)[::-1])[:2*S_omp]               
                    Lambda_sel = Lambda_sel_tmp
                else:
                    Lambda_sel_tmp = (np.argsort(Gmod_bst)[::-1])[:S_chs]
                    Lam_pr_bst  = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/Lam_bst_1dellps_n={N}_genmod_S={S_omp}_{i-1}_j{j}_c{trc}.csv')['Lam_bst'].to_numpy().flatten()
                    print("Lambda_sel_tmp",Lambda_sel_tmp)
                    print("Lambda_pr_bst",Lam_pr_bst)
                    Lambda_sel = np.union1d(Lambda_sel_tmp,Lam_pr_bst)
                    print("Lambda_sel",Lambda_sel)
                
            elif args.add_tpso_res==3: #mixed approach with OMP like approach: don't add more than 4 at a time:
                if i==0:
                    Lambda_sel_tmp = (np.argsort(Gmod_bst)[::-1])[:2*S_omp]               
                    Lambda_sel = Lambda_sel_tmp
                else:
                    c_res = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/comp_rs_1dellps_n={N}_genmod_S={S_omp}_{i-1}_j{j}_c{trc}.csv').to_numpy().flatten()
                    crs_ind_tp = np.argsort(np.abs(c_res))[::-1][:args.sel_res]
                    G_bst_h = Gmod_bst[P_omp:]
                    tpgh_ind = np.argsort(G_bst_h)[::-1][:S_omp] 
                    Lambda_sel_tmp = tpgh_ind + P_omp    
                    Lam_pr_bst  = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/Lam_bst_1dellps_n={N}_genmod_S={S_omp}_{i-1}_j{j}_c{trc}.csv')['Lam_bst'].to_numpy().flatten()
                    Lam_comn1 = np.intersect1d(Lambda_sel_tmp,Lam_pr_bst)
                    print('Lam_comn1',Lam_comn1)
                    Lam_comn2 = np.intersect1d(crs_ind_tp,Lam_pr_bst)
                    print('Lam_comn2',Lam_comn2)
                    Lam_comn = np.union1d(Lam_comn1,Lam_comn2)  
                    print('Lam_comn',Lam_comn)
                    S_comn = Lam_comn.size
                    #S_csit = S_omp+S_chs-S_comn
                    print('tpgh_ind',tpgh_ind)
                    print('G_bst_h',G_bst_h)
                    print('crs_ind_tp',crs_ind_tp)
                    print("Lambda_sel_tmp",Lambda_sel_tmp)
                    print("Lambda_pr_bst",Lam_pr_bst)
                    print("Lam_comn",Lam_comn)

                    if np.size(Lam_comn1) > 0 and np.size(Lam_comn2)==0:
                        Lambda_sel_tmp_g = Lambda_sel_tmp[np.in1d(Lambda_sel_tmp, Lam_comn1, invert=True)][:args.sel_thrd]       
                        Lambda_sel_tmp = np.union1d(Lambda_sel_tmp_g,crs_ind_tp)
                        
                    elif np.size(Lam_comn1) == 0 and np.size(Lam_comn2) > 0:
                        #Lc2_sz = np.size(Lam_comn2)   
                        #add_res_unq = np.setdiff1d(crs_ind_tp,Lam_comn2)
                                #import pdb; pdb.set_trace()
                        Lambda_sel_tmp = np.union1d(Lambda_sel_tmp[:args.sel_thrd],crs_ind_tp) 
                    elif S_comn==0: 
                        Lambda_sel_tmp = np.union1d(Lambda_sel_tmp[:args.sel_thrd],crs_ind_tp) 
                    print("Lambda_sel_tmp",Lambda_sel_tmp)
                    Lambda_sel = np.union1d(Lambda_sel_tmp,Lam_pr_bst)
                    print("Lambda_sel",Lambda_sel)
            elif args.add_tpso_res==4:
                if i==0:
                    Lambda_sel_tmp = (np.argsort(Gmod_bst)[::-1])[:S_chs]               
                    Lambda_sel = Lambda_sel_tmp
        
                else:
                    c_res = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/comp_rs_1dellps_n={N}_genmod_S={S_omp}_{i-1}_j{j}_c{trc}.csv').to_numpy().flatten()
                    crs_ind_tp = np.argsort(np.abs(c_res))[::-1][:args.sel_res]
                    Lambda_sel_tmp = (np.argsort(Gmod_bst)[::-1])[:S_chs]
                    Lam_pr_bst  = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/Lam_bst_1dellps_n={N}_genmod_S={S_omp}_{i-1}_j{j}_c{trc}.csv')['Lam_bst'].to_numpy().flatten()
                    Lam_comn = np.intersect1d(Lambda_sel_tmp,Lam_pr_bst)
                    S_comn = Lam_comn.size
                    S_csit = S_omp+S_chs-S_comn
                    print("Lambda_sel_tmp",Lambda_sel_tmp)
                    print("Lambda_pr_bst",Lam_pr_bst)
                    print("Lam_comn",Lam_comn)

                    if S_csit > S_chs:
                        Lambda_sel_tmp = Lambda_sel_tmp[np.in1d(Lambda_sel_tmp, Lam_comn, invert=True)][:sprsty-args.sel_res]       
                        #import pdb; pdb.set_trace()
                    elif S_comn==0: 
                        Lambda_sel_tmp = Lambda_sel_tmp[:sprsty-args.sel_res]
                    print("Lambda_sel_tmp",Lambda_sel_tmp)
                    Lambda_sel1= np.union1d(Lambda_sel_tmp,Lam_pr_bst)
                    Lambda_sel = np.union1d(Lambda_sel1,crs_ind_tp)
                    print("Lambda_sel",Lambda_sel)
                
            else:

                if i==0:
                    Lambda_sel_tmp = (np.argsort(Gmod_bst)[::-1])[:S_chs]               
                    Lambda_sel = Lambda_sel_tmp
                #elif i==tot_itr-1:
                #    cr_mxind_4 = (np.argsort(np.abs(c_hat))[::-1])[:4]
                #    Lambda_sel_tmp1 = (np.argsort(Gmod_bst)[::-1])[:S_chs-4]
                #    Lambda_sel_tmp = np.union1d(cr_mxind_4,Lambda_sel_tmp1) 
                #    Lam_pr_bst  = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/Lam_bst_1dellps_n={N}_genmod_S={S_omp}_{i-1}_j{j}_c{trc}.csv')['Lam_bst'].to_numpy().flatten()
                #    Lam_comn = np.intersect1d(Lambda_sel_tmp,Lam_pr_bst)
                #    S_comn = Lam_comn.size
                #    S_csit = S_omp+S_chs-S_comn
                #    print("Lambda_sel_tmp",Lambda_sel_tmp)
                #    print("Lambda_pr_bst",Lam_pr_bst)
                #    print("Lam_comn",Lam_comn)

                #    if S_csit > S_chs:
                #        Lambda_sel_tmp = Lambda_sel_tmp[np.in1d(Lambda_sel_tmp, Lam_comn, invert=True)][:sprsty]       
                #        #import pdb; pdb.set_trace()
                #    elif S_comn==0: 
                #        Lambda_sel_tmp = Lambda_sel_tmp[:sprsty]
                #    print("Lambda_sel_tmp",Lambda_sel_tmp)
                #    Lambda_sel = np.union1d(Lambda_sel_tmp,Lam_pr_bst)
                #    print("Lambda_sel",Lambda_sel)
        
                else:
                    Lambda_sel_tmp = (np.argsort(Gmod_bst)[::-1])[:S_chs]
                    Lam_pr_bst  = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/Lam_bst_1dellps_n={N}_genmod_S={S_omp}_{i-1}_j{j}_c{trc}.csv')['Lam_bst'].to_numpy().flatten()
                    Lam_comn = np.intersect1d(Lambda_sel_tmp,Lam_pr_bst)
                    S_comn = Lam_comn.size
                    S_csit = S_omp+S_chs-S_comn
                    print("Lambda_sel_tmp",Lambda_sel_tmp)
                    print("Lambda_pr_bst",Lam_pr_bst)
                    print("Lam_comn",Lam_comn)

                    if S_csit > S_chs:
                        Lambda_sel_tmp = Lambda_sel_tmp[np.in1d(Lambda_sel_tmp, Lam_comn, invert=True)][:sprsty]       
                        #import pdb; pdb.set_trace()
                    elif S_comn==0: 
                        Lambda_sel_tmp = Lambda_sel_tmp[:sprsty]
                    print("Lambda_sel_tmp",Lambda_sel_tmp)
                    Lambda_sel = np.union1d(Lambda_sel_tmp,Lam_pr_bst)
                    print("Lambda_sel",Lambda_sel)
            #import pdb; pdb.set_trace()
            Psi_active_bst = pcu.make_Psi_drn(y_data[optim_indices,:d],mi_mat,Lambda_sel.tolist(),chc_Psi)
            Psi_active_bst_T = np.transpose(Psi_active_bst)
            if args.ls_cv==0:  
                c_hat_bst = (la.inv(Psi_active_bst_T @ Psi_active_bst) @ Psi_active_bst_T @ u_data[optim_indices]).flatten()
                c_omp_sel[Lambda_sel] = c_hat_bst
            else:
                #===============================================================================
                #===============================================================================
                #Least squares with cross validation:    
                
                kf = KFold(n_splits=nfld_ls,shuffle=True,random_state=args.rnd_st_cvls)
                #Psi_active_2nd = pcu.make_Psi_drn(y_data[optim_indices,:d],mi_mat,Lambda_bst.tolist(),chc_Psi)
                #Psi_active_2nd_T = np.transpose(Psi_active_2nd)
                c_omp_bst_fl = np.zeros((P,nfld_ls))
                valid_err_nfld_fl = np.zeros(nfld_ls)
                trn_err_nfld_fl = np.zeros(nfld_ls)
                for i_fld, (trn_ind_fld,tst_ind_fld) in enumerate(kf.split(Psi_active_bst)):
                    #import pdb; pdb.set_trace()
                    c_hat_bst_2nd = (la.inv(Psi_active_bst_T[:,trn_ind_fld] @ Psi_active_bst[trn_ind_fld,:]) @ Psi_active_bst_T[:,trn_ind_fld] @ u_data[optim_indices[trn_ind_fld]]).flatten()
                    c_omp_bst_fl[Lambda_sel.tolist(),i_fld] = c_hat_bst_2nd
                    #data_tst_lscv = {'y_data':y_data[],'u_data':u_data,'val_ind':,'test_ind':test_indices,
                    #            'opt_ind':optim_indices,'Nv':int(nfld_ls*0.2),'chc_poly':chc_Psi,'chc_omp':chc_omp_slv}                 #trn_err_fld, valid_err_fld = tnn.val_test_err(data_tst_lscv,mi_mat,c_omp_bst_fl[:,i_fld])
                    valid_err_fld = la.norm(Psi_active_bst[tst_ind_fld,:] @ c_hat_bst_2nd - u_data[optim_indices[tst_ind_fld]].T) / la.norm(u_data[optim_indices[tst_ind_fld]].T)
                    valid_err_nfld_fl[i_fld] = valid_err_fld
                    trn_err_fld = la.norm(Psi_active_bst[trn_ind_fld,:] @ c_hat_bst_2nd - u_data[optim_indices[trn_ind_fld]].T) / la.norm(u_data[optim_indices[trn_ind_fld]].T)
                    trn_err_nfld_fl[i_fld] = valid_err_fld

                mnfld_ls = np.argmin(valid_err_nfld_fl)
                c_omp_sel = c_omp_bst_fl[:,mnfld_ls]
                #===============================================================================
                #===============================================================================
            #========
            Lambda_bst = (np.argsort(np.abs(c_omp_sel))[::-1])[:S_omp]
            c_omp_bst[Lambda_bst] = c_omp_sel[Lambda_bst]
#FIXME do least squares using Lambda_bst instead of selecting from the already chosen one:  
            #===============================================================================
            #===============================================================================
            #Psi_active_2nd = pcu.make_Psi_drn(y_data[optim_indices,:d],mi_mat,Lambda_sel.tolist(),chc_Psi)
            #Psi_active_2nd_T = np.transpose(Psi_active_2nd)
            #c_hat_bst = (la.inv(Psi_active_2nd_T @ Psi_active_2nd) @ Psi_active_2nd_T @ u_data[optim_indices]).flatten()

            test_err_bst, valid_err_bst = tnn.val_test_err(data_tst,mi_mat,c_omp_bst)
#%%====================calculate the omp coefficients on the residual signal================
            Lambda_bst_mp = [i_mp for i_mp,vl_lmbst in enumerate(Lambda_sel) if vl_lmbst in Lambda_bst]
            print('Lambda_bst',Lambda_bst)
            print('Lambda_sel',Lambda_sel)
            print('Lambda_bst_mp',Lambda_bst_mp)
            #import pdb; pdb.set_trace()            
            rsdl =  u_data[optim_indices] - Psi_active_bst[:,Lambda_bst_mp] @ c_omp_bst[Lambda_sel[Lambda_bst_mp]]
            rsdl_nrm_it.append(la.norm(rsdl))    
#            Psi_omp = pcu.make_Psi(y_data[optim_indices,:d],mi_mat_omp,chc_Psi) 
            #import pdb;pdb.set_trace()
            if args.use_gmd==2:
                df_rsdl_test= pd.DataFrame({'rsdl':rsdl.flatten()})
                df_rsdl_test.to_csv(f'{out_dir_ini}/plots/j={j}/test_rsdl_1dellps_n={N}_genmod_S={S_omp}_j{j}.csv',index=False)
                print("inside genmod loop")
                opt_lst = [*[f'optim.{t_in}' for t_in in range(int(N*4/5))],*[f'valid.{v_in}' for v_in in range(int(N/5))]]
                opt_params_gmd = {'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'stepSize': 0.001,
                              'maxIter': 100000, 'objecTol': 1e-6, 'ALIter': 10,
                              'resultCheckFreq': 10, 'updateLambda': True,
                              'switchSigns': False, 'useLCurve': False, 'showLCurve': False,'Nvlrp': 1}
                df_opt_params_gmd = pd.DataFrame(opt_params_gmd,index=[0])
                df_opt_params_gmd.to_csv(f'{out_dir_ini}/params_genmod_org_adam_N={N}.csv')
                indices_gmd = indices0.copy()
                indices_gmd = indices_gmd.set_axis(opt_lst,axis=1)
                
                #import pdb;pdb.set_trace()
                data_all['u_data'] = rsdl.flatten()  
                #data_all['y_data'] = 
                c_om_rs,Psi_omp =  ro.run_genmod(j, j,'1dellps_gmdorg_n=' + str(N), d, p_0, data_all,indices_gmd,
                      f'{out_dir_ini}/plots', N, Nv, chc_Psi,mi_mat_omp,2*d+1, lasso_eps=1e-10,
                      lasso_iter=1e5, lasso_tol=1e-4, lasso_n_alphas=100,
                      opt_params=opt_params_gmd)
                #df_cgmd = pd.read_csv('../output/titan_ppr/results/csaug13/d=21/p=3/ref_dbg/gmd_org/1dellps_gmdorg_n=100_genmod_kmin=0_1.csv')
                #c_ini = df_cgmd['Coefficients'].to_numpy().flatten()
                #mi_mat_omp = np.copy(mi_mat_p0)
                #P_omp = np.size(mi_mat_omp,0)
                ##S_omp0 = np.nonzero(c_ini) 
                #train_err_p0, valid_err_p0 = tnn.val_test_err(data_tst,mi_mat_omp,c_ini)
            else:
                omp_res = lm.OrthogonalMatchingPursuit(n_nonzero_coefs=S_omp0,fit_intercept=False)
            #    import pdb; pdb.set_trace() 
                print('omp_res:',omp_res)
                omp_res.fit(Psi_omp, rsdl.flatten())
  
                c_om_rs = omp_res.coef_           
#==========================================================================================
# Find the coefficients that lead to minimum validation error 
# and rewrite the coeffs so that the remaining code is undisturbed:  
#            import pdb; pdb.set_trace()
            # eps_c.append(la.norm(c_omp_bst - c_ref)/la.norm(c_ref))
  # if i==0:
#     eps_ctmp[trc] = la.norm(c_omp_bst - c_ref)/la.norm(c_ref)
            # else:
            if chc_eps =='u':
                epsu_tmp[trc] = valid_err_bst
            elif chc_eps =='c':                  
                eps_ctmp[trc] = la.norm(c_omp_bst - c_ref)                
            #trn_ind_nw = np.concatenate((cr_mxind,np.array(best_config.get("tind_nz")),np.array(best_config.get("tind_z"))),axis=None)
            #df_trn_ind_nw = pd.DataFrame({'trn_ind_nw':trn_ind_nw})        
            #df_trn_ind_nw.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/trn_indices_alph_omp_N={N}_{i}_c{trc}.csv',index=False)
            #val_ind_nw = np.setdiff1d(np.linspace(0,P_alg-1,P_alg),trn_ind_nw)
            #df_val_ind_nw = pd.DataFrame({'val_ind_nw':val_ind_nw})        
            #df_val_ind_nw.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/val_indices_alph_omp_N={N}_{i}_c{trc}.csv',index=False)
            
            import pdb; pdb.set_trace()
            trn_ind_nw =  tind_allfld_tpl[best_ind_kcv][0]
            df_trn_ind_nw = pd.DataFrame({'trn_ind_nw':trn_ind_nw})        
            df_trn_ind_nw.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/trn_indices_alph_omp_N={N}_{i}_c{trc}.csv',index=False)
            val_ind_nw = tind_allfld_tpl[best_ind_kcv][1]
            df_val_ind_nw = pd.DataFrame({'val_ind_nw':val_ind_nw})        
            df_val_ind_nw.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/val_indices_alph_omp_N={N}_{i}_c{trc}.csv',index=False)
            #%% Write stuff:
            #df_b_params = pd.DataFrame({key_cfg:val_cfg for key_cfg,val_cfg in best_config.items() if key_cfg not in ['tind_nz','tind_z']},index=[0])
            #df_b_params.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/Best_hyper_params_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)           
            h_prms_dict = {f"a{a_lyr}":avtnlst[a_lyr] for a_lyr in range(len(avtnlst))} 
            h_prms_dict.update({f"h{h_lyr}":hid_layers[h_lyr] for h_lyr in range(len(hid_layers))})
            df_b_params = pd.DataFrame(h_prms_dict,index=[0])
            df_b_params.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/Best_hyper_params_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)           
            df_Lam_sel = pd.DataFrame({'Lam_sel':Lambda_sel})
            df_Lam_sel.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/Lam_sel_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)           
            df_Lam_bst = pd.DataFrame({'Lam_bst':Lambda_bst})
            df_Lam_bst.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/Lam_bst_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)           
            df_c_omp_rs = pd.DataFrame({'comp_rs':c_om_rs})
            df_c_omp_rs.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/comp_rs_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)               
            # cost_tot_1 = cost_tot[1:,:]
            df_bepoch = pd.DataFrame({'ep_best':best_epoch,'kind_cv':best_ind_kcv},index=[0])
            df_bepoch.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/best_epoch_kind_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)
            df_kind = pd.DataFrame({'kind_cv':best_ind_kcv},index=[0])
            df_kind.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/best_kcvind_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)
            df_cost_tot = pd.DataFrame({'cost_t':np.array(cost)})
            df_cost_tot.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/cost_tot_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)
            df_cost_val = pd.DataFrame({'cost_val':np.array(cost_val)})
            df_cost_val.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/cost_val_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)
            # df_cost_val = pd.DataFrame(np.array(cost_val))
            # df_cost_val.to_csv(f'{out_dir_ini}/plots/cost_val_1dellps_n={N}_genmod_S={S_omp}.csv',index=False)
            df_c_hat = pd.DataFrame({'c_hat':c_hat})
            df_c_hat.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/c_hat_tot_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)
            df_Gs = pd.DataFrame({'Gmod_bst':Gmod_bst})
            df_Gs.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/Gmod_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)
            df_c_omp_bst = pd.DataFrame({'comp_fnl':c_omp_bst})
            df_c_omp_bst.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/comp_fl_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)
            df_c_omp_sel = pd.DataFrame({'comp_sel':c_omp_sel})
            df_c_omp_sel.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/comp_sel_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)
            df_cini = pd.DataFrame({'cini':c_ini})
            df_cini.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/cini_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)
            # df_eps_ctmp = pd.DataFrame({'eps_ctmp':eps_ctmp}) 
            # df_eps_ctmp.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/eps_ctmp_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)
            df_thet_up = pd.DataFrame({'thet_bst':thet_bst.detach().numpy()})
            df_thet_up.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/thetup_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)  
            #import pdb; pdb.set_trace()
            df_thet_dict = pd.DataFrame(thet_dict_full[best_ind_kcv])
            df_thet_dict.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/thet_hist_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)  
          #df_thet_up = pd.DataFrame({'thet_f':thet_f,'thet_bst':thet_bst})
          #df_thet_up.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/thetup_1dellps_n={N}_genmod_S={S_omp}_{i}_j{j}_c{trc}.csv',index=False)  

        if chc_eps == 'c':
            ecmn_ind[i] = np.argmin(eps_ctmp)
            eps_c[i+1] = eps_ctmp[int(ecmn_ind[i])]
        elif chc_eps == 'u': 
            ecmn_ind[i] = np.argmin(epsu_tmp)
            eps_u[i+1] = epsu_tmp[int(ecmn_ind[i])]            
    
        # # Testing error: is the error on the training data:
        if la.norm(rsdl) < args.resomp_tol:
            break
        i += 1
        # #=================================================================
    #%% Plot relative validation error:
    # Wrtite relative coefficient error:
    if chc_eps == 'u':
        df_epsu = pd.DataFrame({'eps_u':eps_u})
        df_epsu.to_csv(f'{out_dir_ini}/plots/j={j}/epsu_1dellps_n={N}_genmod_S={S_omp}_j{j}.csv',index=False)  
        print(f'relative validation error for the iteration-{i}:',valid_err_bst)
        # Wrtite relative coefficient error:
    elif chc_eps == 'c': 
        df_epsc = pd.DataFrame({'eps_c':eps_c})
        df_epsc.to_csv(f'{out_dir_ini}/plots/j={j}/epsc_1dellps_n={N}_genmod_S={S_omp}_j{j}.csv',index=False)        
    df_mnd= pd.DataFrame({'ecmn_ind':ecmn_ind})
    df_mnd.to_csv(f'{out_dir_ini}/plots/j={j}/ecmn_ind_1dellps_n={N}_genmod_S={S_omp}_j{j}.csv',index=False)
    df_rsdl= pd.DataFrame({'rsdl':np.array(rsdl_nrm_it)})
    df_rsdl.to_csv(f'{out_dir_ini}/plots/j={j}/rsdl_1dellps_n={N}_genmod_S={S_omp}_j{j}.csv',index=False)
    # Wrtite weighted coefficient error:
    df_epsc_abs = pd.DataFrame({'epsc_abs':eps_abs})
    df_epsc_abs.to_csv(f'{out_dir_ini}/plots/j={j}/epsc_abs_1dellps_n={N}_genmod_S={S_omp}_j{j}.csv',index=False)        
    if chc_eps=='c':
        df_cref = pd.DataFrame({'c_ref':c_ref})
        df_cref.to_csv(f'{out_dir_ini}/plots/j={j}/c_ref_1dellps_n={N}_genmod_S={S_omp}_j{j}.csv',index=False)
    #df_epscomp = pd.DataFrame({'epsu_omph':valid_omp_ph,'epsu_omph_t':test_omp_ph},index=[0])
    #df_epscomp.to_csv(f'{out_dir_ini}/plots/epsuomph_tst_1dellps_n={N}_genmod_S={S_omp}_j{j}.csv',index=False)
#plt.show()



#

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:08:53 2023

@author: jothi
"""
import spgl1
import scipy.io as sio
from scipy.stats import norm
import seaborn as sns
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import csv
import random
import sklearn.linear_model as lm
import sys
import os
#sys.path.append('C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod_omp')
sys.path.append('/home/jothi/CoSaMP_genNN')
out_dir_ini = "../output/titan_ppr/results/csaug13/d=21/p=3/spgl1/test"
out_dir_sct = '../data/Titn_rcnt_dta/d21'
import genmod_mod.polynomial_chaos_utils as pcu
import genmod_mod.train_NN_omp_wptmg as tnn
#prompt for asking if the output directory should be created:
#import pdb; pdb.set_trace()
if not os.path.exists(f"{out_dir_ini}"):
    os.makedirs(f"{out_dir_ini}")
else:
    print("Error: The output directory already exists-")
    print("press any key continue/Ctrl+z to exit")
    input()
    
    #sys.exit() #FIXME

#parameters:
d = 21
p = 3
N_y = 2000+6000 #FIXME
Chc_Psi = 'Hermite' # 'Hermite' or 'Legendre'.
#y_data1 = pd.read_csv(f'{out_dir_sct}/ydt_stdnrm_ttn_Aval_ln_pd8th.csv').to_numpy()
#u_data2 = pd.read_csv(f'{out_dir_sct}/xCN_smp.csv').to_numpy()
#u_data1 = np.transpose(u_data2[:,2:])
#Read data for the 20-dimensional problem:
#y_data1 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N1000_d20.csv').to_numpy()
#u_data1 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_PLATO_to_MURP_N1000.csv').to_numpy()
#Read data for the 21-dimension problem:
y_data11 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N1000_d21_sd100_plsU4_CH4.csv').to_numpy()
y_data12 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N1000_d21_sd200_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
y_data13 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N8000_d21_sd300_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
y_data1 = np.vstack((y_data11,y_data12,y_data13))
#import pdb; pdb.set_trace()
y_data1[:,d-1] = norm.ppf(0.5*(y_data1[:,d-1]+1))
u_data11 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_PLATO_to_MURP_N1000.csv').to_numpy()
u_data12 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N1000_ext1k.csv').to_numpy()
u_data13 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N6000.csv').to_numpy()
u_data1 = np.hstack((u_data11,u_data12,u_data13))
x_data11 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000.csv').to_numpy()
x_data12 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000_ext1k.csv').to_numpy()
x_data13 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_Plato_to_MURP_N6000.csv').to_numpy()
x_data = np.hstack((x_data11,x_data12,x_data13))
u_data = np.mean(u_data1,axis=0)
y_data = y_data1[:N_y,0:d] #FIXME
#u_data = np.amax(u_data1,axis=1)
mi_mat = pcu.make_mi_mat(d, p)
P = np.size(mi_mat,0)   
#%% Visualize udata:
plt.figure(21)
#u_plt_rnd = random.sample(range(0,np.size(u_data)), 20)
#u_plt_rnd = random.sample(range(2000,np.size(u_data)), N_y-2000) #FIXME.
u_plt_rnd = random.sample(range(2000),2000) #FIXME.
#import pdb; pdb.set_trace()
for i in u_plt_rnd:
    plt.plot(x_data[:,i],u_data1[:,i])
    # plt.plot(u_data1[i,:],label=f'i={i}')
    #plt.plot(u_data1[:,i],label=f'i={i}')
    plt.xlabel('spatial location x')    
    plt.ylabel('Radiative heat flux[W/cm^2]')    
plt.savefig(f"{out_dir_ini}/u_data.png",dpi=300)
#%% Visualize ydata:
#==============================================================================
# Plot scatter plot of input parameters (Jacqui)
plt.figure(1)
y_df = pd.DataFrame(y_data)
y_df.shape
sns.pairplot(y_df.iloc[:, :d], kind="scatter")
plt.savefig(f"{out_dir_ini}/pair_plot_y.png") #,dpi=300)
#import pdb; pdb.set_trace()
#for i in range(d):    
#    # plt.figure(i)
#    if i<4:
#        plt.figure(1)
#        plt.subplot(2,2,i+1)
#    elif i>=4 and i <8:    
#        plt.figure(2)
#        plt.subplot(2,2,i+1-4)
#    elif i>=8 and i <12:
#        plt.figure(3)
#        plt.subplot(2,2,i+1-8)
#    else:
#        plt.figure(4)
#        plt.subplot(2,2,i+1-12)
#    n, count, patches = plt.hist(y_data[:,i],100, density=False, facecolor='g') 
#    # n, count, patches = plt.hist(Aval[:,i],100, density=False, facecolor='g')
#    # n1, count1, patches1 = plt.hist(Aval_uni[:,i],100, density=False, facecolor='b')
#    plt.xlabel(f'y{i}')    
#    # plt.ylabel('pdf')
#    if i==0 or i==8:
#        plt.ylabel('Frequency')    
#    # plt.axvline(x=ref_Asmp[i],color='r')    
#    plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
#    plt.tight_layout()    
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels  
#%% Validation error function:
def val_train_err(data_tst,mi_mat_t,c):
    y_data  = data_tst['y_data']
    u_data  = data_tst['u_data']   
    test_indices = data_tst['test_ind']
    train_indices  = data_tst['train_ind']
    valid_indices  = data_tst['val_ind']
    Nv = data_tst['Nv']
    chc_Psi = data_tst['chc_poly'] 
    d = np.size(mi_mat_t,1)
    if chc_Psi == 'Legendre':
        Psi_train = pcu.make_Psi(y_data[train_indices, :d], mi_mat_t,chc_Psi)
        # trn_err = la.norm(
        #     Psi_train @ c - u_data[train_indices].T
        # ) / la.norm(u_data[train_indices].T)
        trn_err = la.norm(
            Psi_train @ c - u_data[train_indices].T
        ) 
        # Validation error: is the error on the unseen data:
        Psi_valid = pcu.make_Psi(y_data[valid_indices[:Nv], :d], mi_mat_t,chc_Psi)
        # valid_err = la.norm(
        #     Psi_valid @ c - u_data[valid_indices[:Nv]].T
        # ) / la.norm(u_data[valid_indices[:Nv]].T)
        valid_err = la.norm(
            Psi_valid @ c - u_data[valid_indices[:Nv]].T
        )
        # Psi_tst = pcu.make_Psi(y_data[test_indices[:Nts], :d], mi_mat_t,chc_Psi)
        # test_err = la.norm(
        #     Psi_tst @ c - u_data[test_indices[:Nts]].T
        # )
    elif chc_Psi == 'Hermite':
        Psi_train = pcu.make_Psi(y_data[train_indices, :d], mi_mat_t,chc_Psi)
        # trn_err = la.norm(
        #     Psi_train @ c - u_data[train_indices].T
        # ) / la.norm(u_data[train_indices].T)
        trn_err = la.norm(
            Psi_train @ c - u_data[train_indices].T
        ) 
        # Validation error: is the error on the unseen data:
        Psi_valid = pcu.make_Psi(y_data[valid_indices[:Nv], :d], mi_mat_t,chc_Psi)
        # valid_err = la.norm(
        #     Psi_valid @ c - u_data[valid_indices[:Nv]].T
        # ) / la.norm(u_data[valid_indices[:Nv]].T)
        valid_err = la.norm(
            Psi_valid @ c - u_data[valid_indices[:Nv]].T
        )
        # Psi_tst = pcu.make_Psi(y_data[test_indices[:Nts], :d], mi_mat_t,chc_Psi)
        # test_err = la.norm(
        #     Psi_tst @ c - u_data[test_indices[:Nts]].T
        # )
        
    return trn_err, valid_err

#%% Write index file:
Nrep = 1
N = 4000  # note that you need more samples than sparsity for least squares.
iter_max = 5000
seed_ind =1
Nts = 4000 #Number of testing points.
Nv = int(N / 5) #validation points for cross validation (basically part of the training)
#set the SPGL1 optimization parameters:
optTol = 1e-9
bpTol = 1e-9
lsTol = 1e-9
f = open(f'{out_dir_ini}/1dellps_indices_n={N}.csv', 'w')
N_tot = np.size(u_data)
# N_tot = np.size(y_data,0)
random.seed(seed_ind) 
fw = csv.writer(f)
header =[*["optim"] * (int(N * 4 / 5)), *["valid"] * (int(N / 5))]
np.size(header)
fw.writerow(header)
for i in range(50):
    fw.writerow(random.sample(range(N_tot), N))
f.close()
index_file = f'{out_dir_ini}/1dellps_indices_n={N}.csv'
# index_file = f'{out_dir_ini}/indices/1dellps_indices_n=100.csv'
indices0 = pd.read_csv(index_file)
sigma_lst1 = np.arange(0.0001,0.001,0.0001)
sigma_lst2 = np.arange(0.002,0.008,0.001)
sigma_lst = np.concatenate((sigma_lst1,sigma_lst2)) 
sigma_lst = np.arange(0.001,0.006,0.001) #FIXME 
del_r_Nrp = np.zeros(len(sigma_lst))
del_v_Nrp = np.zeros_like(del_r_Nrp)
del_cpv_ind = []
mi_mat_spg = pcu.make_mi_mat(d, p)
S_omph_cv = []
#create dataframe and store the input values:
df_params = pd.DataFrame({'p':p,'d':d,'N':N,'Nrep':Nrep,'it_mx':iter_max,'Nts':Nts,'Nv':Nv,'Sig_rng':sigma_lst,'Chc_Psi':Chc_Psi,'seed_ind':seed_ind})
df_params.to_csv(f"{out_dir_ini}/params_SPGL1_N{N}.csv")
#%% Cross validation procedure for selecting optimal S value:    
optim_indices = indices0.iloc[0].to_numpy()
test_indices = np.setdiff1d(range(np.size(u_data)), optim_indices)
valids = [name for name in indices0.columns if name.startswith("valid")]
valid_indices = indices0.loc[0][valids].to_numpy() 
trains = [name for name in indices0.columns if name.startswith("optim")]
train_indices = indices0.loc[0][trains].to_numpy() 
data_tst = {'y_data':y_data,'u_data':u_data,'val_ind':valid_indices,'train_ind':train_indices, 
            'test_ind':test_indices,'Nv':Nv,'chc_poly':Chc_Psi} 
P_spg = np.size(mi_mat_spg,0)
Psi_spg = pcu.make_Psi(y_data[train_indices,:d],mi_mat_spg,Chc_Psi)
del_r = []
del_v = []
c_spg_hst = np.zeros((P_spg,len(sigma_lst)))
ct = 0
#import pdb; pdb.set_trace()
for sigma in sigma_lst:
    #Optimum Indices:
    c_spg, resid, grad, info = spgl1.spg_bpdn(Psi_spg, u_data[train_indices], 
                                          sigma, iter_lim=iter_max, verbosity=1,opt_tol= optTol,
                                          bp_tol = bpTol, ls_tol = lsTol)
    #import pdb; pdb.set_trace()
    S_spg = np.size(np.nonzero(c_spg)[0])
    c_spg_hst[:,ct] = c_spg
    ## Be careful when you use \Psi for testing:
    train_omp_ph, valid_omp_ph = val_train_err(data_tst,mi_mat_spg,c_spg)
    del_r.append(train_omp_ph)
    del_v.append(valid_omp_ph)
    ct += 1
del_v_np = np.array(del_v)
dvmn_ind = np.argmin(del_v_np)
delr_min = del_r[dvmn_ind]
df_delv = pd.DataFrame({'del_v':del_v})     
df_delv.to_csv(f'{out_dir_ini}/del_v_p={p}_d={d}_crsvld_spgl1.csv',index=False)
df_delr = pd.DataFrame({'del_r':del_r})
df_delr.to_csv(f'{out_dir_ini}/del_r_p={p}_d={d}_crsvld_spgl1.csv',index=False)

#%% Test it on the unseen test data:
# sigma_opt = np.sqrt(N/Nv) * delr_min
sigma_opt = delr_min
df_delv_min = pd.DataFrame({'delv_min':del_v_np[dvmn_ind],'delr_min':delr_min,'sig_opt':sigma_opt},index=[0])
df_delv_min.to_csv(f'{out_dir_ini}/delvr_min_p={p}_d={d}_crsvld_spgl1.csv')
Psi_trn_opt = pcu.make_Psi(y_data[optim_indices,:d],mi_mat_spg,Chc_Psi)
c_opt, resid, grad, info = spgl1.spg_bpdn(Psi_trn_opt, u_data[optim_indices], 
                                      sigma_opt, iter_lim=iter_max, verbosity=2,opt_tol= optTol,
                                      bp_tol = bpTol, ls_tol = lsTol)
S_opt_spg = np.size(np.nonzero(c_opt))
#import pdb; pdb.set_trace()
optim_indices = indices0.iloc[0].to_numpy()
# optim_indices_fl = indices0.iloc[0:10].to_numpy().flatten()
test_indices = np.setdiff1d(range(np.size(u_data)), optim_indices)
Psi_tst = pcu.make_Psi(y_data[test_indices[:Nts], :d], mi_mat_spg,Chc_Psi)
test_err = la.norm(
    Psi_tst @ c_opt - u_data[test_indices[:Nts]].T
) / la.norm(u_data[test_indices[:Nts]].T)
df_delv = pd.DataFrame({'test_err':test_err},index=[0])     
df_delv.to_csv(f'{out_dir_ini}/test_err_p={p}_d={d}_crsvld_spgl1.csv')
df_copt = pd.DataFrame({'c_opt':c_opt})     
df_copt.to_csv(f'{out_dir_ini}/c_opt_p={p}_d={d}_crsvld_spgl1_S={S_opt_spg}.csv',index=False)
#%% plot figure:
plt.figure(100)
plt.plot(sigma_lst,del_v,'--*r',label='$\delta_v$')
plt.plot(sigma_lst,del_r,'--*b',label='$\delta_r$')
plt.xlabel('$\sigma$')
plt.title(f'$p={p},d={d},\epsilon_u(test)={round(test_err,6)}$')
# plt.ylabel('$\delta$')  
plt.legend()  
plt.savefig(f'{out_dir_ini}/delv_vs_r_p={p}_d={d}.png')
import pdb; pdb.set_trace()

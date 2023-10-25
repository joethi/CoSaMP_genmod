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
import ray
import argparse
import os
# from numpy import log10, sign, abs
np.random.seed(1)
# torch.manual_seed(0)
sys.path.append('/home/jothi/CoSaMP_genNN')
#import pdb;pdb.set_trace()
sys.path.append('/home/jothi/CoSaMP_genNN/scripts/GenMod-org-Hmt')
#sys.path.append('/home/jothi/GenMod_omp/scikit-learn-main/sklearn/linear_model')
# sys.path.append('C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod_omp')
# out_dir_ini = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod_omp/output/duff_osc_ppr'
# out_dir_ini = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod_omp/output/wing_wght'
# out_dir_ini = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod_omp/output/1DElliptic_ppr'
#out_dir_ini = f'../output/titan_ppr/results'
# Redirect stdout to a file
#sys.stdout = open(f'{out_dir_ini}/plots/log_printed.txt', 'w')
import genmod.run_optimizations as ro
import genmod_mod_test.polynomial_chaos_utils as pcu
import genmod_mod_test.Gmodel_NN as gnn
import genmod_mod_test.train_NN_omp_wptmg_test as tnn
import genmod_mod_test.omp_utils as omu
import warnings
#import omp as omp1
#from genmod_mod import Gmodel_NN as gnn_test
#import _omp as omp_scd
#import pdb;pdb.set_trace()
parser = argparse.ArgumentParser(description='Parse the inputs such as output directory name, number of neurons, and so on...')
parser.add_argument('--out_dir_ini',dest = 'out_dir_prs', type=str, help='specify out_dir_ini directory-it is the location of outputs similar to f\'../output/titan_ppr/results\'')
#parser.add_argument('--H',dest='N_Hid',default=10,type=int,help='number of hidden neurons')
parser.add_argument('--S0',dest='S_0',default=9,type=int,help='Sparsity for p0')
parser.add_argument('--Sh',dest='S_h',default=7,type=int,help='Sparsity for ph')
parser.add_argument('--N',dest='N_smp',default=100,type=int,help='Number of samples')
parser.add_argument('--Nv',dest='N_v',default=4000,type=int,help='Number of validation samples')
parser.add_argument('--Nt',dest='N_t',default=1,type=int,help='Number of total iterations')
parser.add_argument('--Nrep',dest='N_rep',default=1,type=int,help='Number of sample replications')
parser.add_argument('--Nlhid',dest='N_hid',default=1,type=int,help='Number of sample replications')
parser.add_argument('--epochs',dest='ep',default=15000,type=int,help='Number of epochs')
parser.add_argument('--iter_fix',dest='it_fix',default=9999,type=int,help='After this number the metric is evaluated')
parser.add_argument('--p_h',dest='ph',default=3,type=int,help='p_h')
parser.add_argument('--hbd',dest='h_bnd',nargs='+',default=[1,7],type=int,help='p_h')
parser.add_argument('--p_0',dest='p0',default=2,type=int,help='p_0')
parser.add_argument('--top_i1',dest='topi1',default=2,type=int,help='p_0')
parser.add_argument('--top_i0',dest='topi0',default=3,type=int,help='p_0')
parser.add_argument('--vlcf_add',dest='vlcfadd',default=0,type=int,help='p_0')
parser.add_argument('--d',dest='dim',default=21,type=int,help='d')
parser.add_argument('--ompsol',dest='chc_omp',default='stdomp',type=str,help='p_h')
parser.add_argument('--poly',dest='chc_poly',default='Hermite',type=str,help='p_h')
parser.add_argument('--use_gmd',dest='use_gmd',default=0,type=int,help='d')
parser.add_argument('--cini_fl',dest='cht_ini_fl',default='/home/jothi/CoSaMP_genNN/output/titan_ppr/results/csaug13/d=21/p=3/ref_dbg/ttne_913_j2smx2S_Nt5/plots/j=1/it=3/c_hat_tot_1dellps_n=100_genmod_S=7_3_j1_c0.csv',type=str,help='file name with the path for initial omp coefficients for reproducing/debugging')
parser.add_argument('--ntrial',dest='num_trl',default=10,type=int,help='num_trials')
parser.add_argument('--dbg',dest='debug_alg',default=0,type=int,help='1-flag for switching to debugging')
parser.add_argument('--j_rng',dest='j_flg',nargs='+',default=0,type=int,help='0 if all the repications needed, a list having necessary replication numbers otherwise')
parser.add_argument('--plot_dat',dest='plot_dat',default=0,type=int,help='plot u data?')
parser.add_argument('--ts',dest='tune_sig',default=1,type=int,help='tune signal-0 is for debugging single layer network-errors out when you use 2 layers and ts 0 concurrently')
args = parser.parse_args()
#test Ray remote function:
#print(sys.path)
#print("---------before ray-------------")
#@ray.remote
#def f():
#    # Print the PYTHONPATH on the worker process.
#    import sys
#    print(sys.path)
#
#f.remote()
#import pdb; pdb.set_trace()
#********RESTART KERNEL IF SWITCHING BW DIFFERENT FOLDERS**********
#==============DO NOT USE THIS CODE TO USE Ncrp>1==================
# Set parameters
p = args.ph 
p_0 = args.p0
d = args.dim  # Set smaller value of d for code to run faster
S_omp = args.S_h
S_omp0 = args.S_0
num_trial = args.num_trl
S_chs = 2*S_omp
freq = 1 
W_fac = [1.0,1.0,1.0,1.0,1.0]
sprsty = S_omp
chc_eps = 'u'
chc_Psi = args.chc_poly #'Hermite'
chc_omp_slv= args.chc_omp #'stdomp'#'ompcv' #'stdomp' #FIXME  
tune_sg = args.tune_sig
pltdta = args.plot_dat #switch to 1 if data should be plotted.
top_i1 = args.topi1 #int(4*cini_nz_ln/5-ntpk_cr), ntpk_cr = top_i1. 4*cini_nz_ln/5 should be > ntpk_cr
top_i0 = args.topi0 
#Seed values:
seed_ind = 1
seed_thtini = 1
sd_thtini_2nd = 3
seed_ceff = 2
random.seed(seed_ind) # FIXME set seeding for reproducibility/debugging purposes.
#Hid = args.N_Hid # number of neurons
Nlhid = args.N_hid
hid_layers = [tune.randint(args.h_bnd[0],args.h_bnd[1]) for __ in range(Nlhid)] 
GNNmod_ini = gnn.GenNN([d] + [hid_layers[hly].sample() for hly in range(len(hid_layers))] +[1])
z_n = sum(prm_NN.numel() for prm_NN in GNNmod_ini.state_dict().values())

#import pdb; pdb.set_trace()
#Hid = hid_layers[:]
#z_n = d * Hid  + 2*Hid + 1    
out_dir_ini = args.out_dir_prs
#import pdb; pdb.set_trace()
start_time = time.time()
print('start and start time:',start_time,start_time)
if os.path.exists(f'{out_dir_ini}/plots'):
    print(f"{out_dir_ini}/plots already exists- Do you want to remove directory (Y/n)")
    Answer = input()
    if Answer=="Y":
        os.system(f'rm -r {out_dir_ini}/plots')
    else:
        print("Directory exists already-exiting to prevent overwriting---")
        sys.exit()
os.makedirs(f'{out_dir_ini}/plots')
#%% Load data
## CHANGE THE VALIDATION ERROR FUNCTION FOR ELLIPTIC EQUATION AS \PSI IS FROM LEGENDRE POLY:
# data = sio.loadmat('C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod_omp/data/dataset1/1Dellptic_data.mat')
# c_dt = sio.loadmat('C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod_omp/data/dataset1/cref_py.mat')
# c_ref = c_dt['cref_py'].flatten()
c_ref = []
# cr_mxind = (np.argsort(np.abs(c_ref))[::-1])[:10]
#Normalize the Avalues:
#%% Normalize the lognormal data (aval) to std gaussian:
#Aval = pd.read_csv(f'{out_dir_sct}/lnpdst_N30k_dx3/files/Aval_sipd_inc8.csv').to_numpy()
##print(np.shape(Aval))
#d=16
#std_lgA = np.zeros(d)
#mean_lgA = np.zeros(d)
#ydat_stnrm = np.zeros_like(Aval)
#for i in range(d):    
#     nrm_y = np.log(Aval[:,i])
#     std_lgA[i] = sts.stdev(nrm_y)
#     mean_lgA[i] = sts.mean(nrm_y)
#
#     ydat_stnrm[:,i] = (nrm_y-mean_lgA[i])/std_lgA[i]
#     x = np.linspace(-4.0,4,1000)
#     plt.figure(202)
#     plt.subplot(4,4,i+1)
#     n, count, patches = plt.hist(ydat_stnrm[:,i],100, density=True, facecolor='g',label=r'$(ln(A)-\mu)/(\sigma)$')
#     plt.plot(x,norm.pdf(x,0,1),color='r',label='N(0,1)')
#     plt.xlabel(f'y{i}')
#     if i==0:
#         plt.ylabel('pdf')
#plt.tight_layout()     
#plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
#plt.savefig(f'{out_dir_sct}/lnpdst_N30k_dx3/figures/ynrm_ln.png')     
#df_ydtnrm = pd.DataFrame(ydat_stnrm)    
#df_ydtnrm.to_csv(f'{out_dir_sct}/lnpdst_N30k_dx3/ydt_stdnrm_ttn_Aval_ln_pd8th.csv',index=False)
#import pdb; pdb.set_trace()
#titan:
#===========================================================================
#===========================================================================
# Read for the Maximum CN concentration for the previous data:
#out_dir_sct = '../data/Titn_rcnt_dta/d21'    
out_dir_sct = '../data/Titn_rcnt_dta/dx1em3/LN_d78_hghunkF'    
#y_data1 = pd.read_csv(f'{out_dir_sct}/files/Aval_zis_frm_invmp_78d_N20k.csv').to_numpy() #incorrect file
y_data1 = pd.read_csv(f'{out_dir_sct}/files/Aval_zis_frm_invmp_78_ln_hghunkF.csv').to_numpy()
y_data = y_data1[:,0:d] 
u_data2 = pd.read_csv(f'{out_dir_sct}/xCN/xCN_smp.csv').to_numpy()
u_data1 = np.transpose(u_data2[:,2:]) #First two colums are just junks.
u_data = np.amax(u_data1,axis=1)
#===========================================================================
#===========================================================================
#===========================================================================
#===========================================================================
# read data for MURP outputs: 
##u_data1 has a shape of "Number of grid points by N_samp" for Wall-directed radiative heat flux.
#y_data11 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N1000_d21_sd100_plsU4_CH4.csv').to_numpy()
#y_data12 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N1000_d21_sd200_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
#y_data13 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N8000_d21_sd300_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
#y_data13 = y_data13[:6000,:d] #FIXME
#y_data14 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N4000_d21_sd400_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
#y_data15 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N2000_d21_sd500_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
#y_data1 = np.vstack((y_data11,y_data12,y_data13,y_data14,y_data15))
##y_data1 = np.vstack((y_data11,y_data12,y_data13)) #FIXME
##===========================================================================
##===========================================================================
##import pdb; pdb.set_trace()
#y_data1[:,d-1] = norm.ppf(0.5*(y_data1[:,d-1]+1))
#u_data11 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_PLATO_to_MURP_N1000.csv').to_numpy()
#u_data12 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N1000_ext1k.csv').to_numpy()
#u_data13 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N6000.csv').to_numpy()
#u_data14 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N6000_ext6k_4kp2k.csv').to_numpy()
#u_data1 = np.hstack((u_data11,u_data12,u_data13,u_data14))
##u_data1 = np.hstack((u_data11,u_data12,u_data13)) #FIXME
#x_data11 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000.csv').to_numpy()
#x_data12 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000_ext1k.csv').to_numpy()
#x_data13 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_Plato_to_MURP_N6000.csv').to_numpy()
#x_data14 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_Plato_to_MURP_N6000_ext6k_4kp2k.csv').to_numpy()
##x_data = np.hstack((x_data11,x_data12,x_data13)) #FIXME
#x_data = np.hstack((x_data11,x_data12,x_data13,x_data14))
#u_data = np.mean(u_data1,axis=0)
#y_data = y_data1[:,0:d] 
#===========================================================================
#===========================================================================
import pdb;pdb.set_trace()
print('u_data1[:5,:5]',u_data1[:5,:5])
print('shape of u_data1:',u_data1.shape)
print('shape of u_data:',u_data.shape)
print('shape of y_data:',y_data.shape)
#%%
#%% Read data:
# Elliptic:
# y_data1 = data['y']
# y_data = y_data1[:,0:d]
# u_data = data['u'].flatten() 
# Titan:
#import pdb;pdb.set_trace()
Nsmp_tot = np.size(u_data)
#%% Visualize the data:
# plt.figure(20)
# # plt.subplot(221)
# plt.hist(y_data[:,0],label = '$y_1$')
# plt.legend()
#==============================================================================
#%% Visualize udata:
#==============================================================================
if pltdta:
    plt.figure(1)
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
    #==============================================================================
    # Plot scatter plot of input parameters (Jacqui)
    plt.figure(2)
    y_df = pd.DataFrame(y_data)
    y_df.shape
    sns.pairplot(y_df.iloc[:, :d], kind="scatter")
    plt.savefig(f"{out_dir_ini}/pair_plot_y.png") #,dpi=300)
#==============================================================================
#==============================================================================
#import pdb;pdb.set_trace()
#
data_all = {'y_data':y_data,'u_data':u_data} 
mi_mat = pcu.make_mi_mat(d, p)
mi_mat_p0 = pcu.make_mi_mat(d, p_0)
df_mimat = pd.DataFrame(mi_mat)
df_mimat.to_csv(f'{out_dir_ini}/mi_mat_pd={p,d}.csv',index=False)
P = np.size(mi_mat,0)  
#print('n',n)
#%% initial parameters:
# sprsty = 43
tot_itr = args.N_t
Nc_rp = 1 # NOTE: Set this as always 1 for this particular case.
Nrp_vl = 1
ecmn_ind = 0
# sprsty = 5  #Think about giving sparsity=1, some matrix manipulations might get affected.
learning_rate = 0.001
epochs = args.ep
#avtnlst =['None' for a_m in range(Nlhid)]#[nn.Sigmoid()] # for the final layer by default exp decay is enforced, so the size is number of layers-1.
avtnlst =['None'  if tune_sg==0 else tune.choice(['None',nn.Sigmoid(),nn.ReLU()]) for a_m in range(Nlhid)] #[nn.Sigmoid()] # for the final layer by default exp decay is enforced, so the size is number of layers-1.

#import pdb; pdb.set_trace()
#avtnlst = ['None',nn.Sigmoid(),nn.Relu()]
#%% Write index file:
N = args.N_smp  # note that you need more samples than sparsity for least squares.
Nv = args.N_v
Nrep = args.N_rep
j_rng = range(Nrep) if args.j_flg==0 else args.j_flg #range(Nrep) ---change this to run for a particular replication. Useful for debugging.
#% Save parameters:
opt_params = {'ph':p,'p0':p_0,'d':d,'epochs':epochs,'lr':learning_rate,'Sh':sprsty,'S0':S_omp0,
        'N_t':tot_itr,'fr':freq,'W_fac':f'{W_fac}','z_n':z_n,'Tp_i1':top_i1,'Tp_i0':top_i0,'N':N,'Nv':Nv,'Nrep':Nrep,'Nc_rp':Nc_rp,'S_chs':S_chs,'chc_poly':chc_Psi,'sd_ind':seed_ind,'sd_thtini':seed_thtini,'sd_ceff':seed_ceff,'Nrp_vl':Nrp_vl,"sd_thtini_2nd":sd_thtini_2nd,'iter_fix':args.it_fix}
#import pdb;pdb.set_trace() 
df_params = pd.DataFrame(opt_params,index=[0])
df_params.to_csv(f'{out_dir_ini}/plots/params_genmod_omp_N={N}_ini.csv')
print(df_params)
f = open(f'{out_dir_ini}/plots/1dellps_indices_n={N}_ini.csv', 'w')
N_rndsmp = np.size(u_data)
fw = csv.writer(f)
header =[*["optim"] * (int(N * 4 / 5)), *["optim"] * (int(N / 5))]
np.size(header)
fw.writerow(header)
#random.seed(seed_ind) # set seeding for reproducibility/debugging purposes#FIXME seed place..
for i in range(Nrep):
 fw.writerow(random.sample(range(N_rndsmp), N))
f.close()
# FIXME: HARDCODING:
index_file = f'{out_dir_ini}/plots/1dellps_indices_n={N}_ini.csv'
#index_file =f'/home/jothi/CoSaMP_genNN/output/titan_ppr/results/csaug13/d=21/p=3/ref_data/1dellps_indices_n=100_ini.csv' 
#print("=====================================================")
#print("Here I hardcoded the indices to 100--please change it")
#print("=====================================================")
#index_file = f'{out_dir_ini}/ini/ind/indices_CoSaMP_N=100_Jul15.csv'
#
indices0 = pd.read_csv(index_file)
df_indc0 = pd.DataFrame(indices0)
df_indc0.to_csv(f'{out_dir_ini}/plots/indices_genmod_omp_N={N}.csv',index=False)
print('shape of indices0:',indices0.shape)
epsc_omph = []
eps_c_omp_abs = []
eps_abs = []
eps_c = np.zeros(tot_itr+1)
eps_u = np.zeros(tot_itr+1)
#import pdb;pdb.set_trace()
#=================================================================================
#=============Test CoSaMP cv for 10 sample replications==========================
#=================================================================================
#for j in j_rng:
#    print(f'=============#replication={j}============')
#    ecmn_ind = np.zeros(tot_itr)
#    os.makedirs(f'{out_dir_ini}/plots/j={j}',exist_ok=True)
#    optim_indices = indices0.iloc[j].to_numpy()
#    valid_indices = np.setdiff1d(range(np.size(u_data)), optim_indices)
#    trains = [name for name in indices0.columns if name.startswith("optim")]
#    test_indices = indices0.loc[j][trains].to_numpy()
##    import pdb; pdb.set_trace()
#    data_tst = {'y_data':y_data,'u_data':u_data,'val_ind':valid_indices,'test_ind':test_indices,'opt_ind':optim_indices,'Nv':Nv,
#            'chc_poly':chc_Psi,'chc_omp':chc_omp_slv} 
#    #
#    Psi_csmp = pcu.make_Psi(y_data[optim_indices,:d],mi_mat,chc_Psi)
#    csmp_prms = {'maxit_csmp':10,'hlcrt_csmp':'iter','tolres_csmp':1e-2,'rnd_st_cvcs': 1} #'iter'
#    df_csmp_prms = pd.DataFrame(csmp_prms,index=[0])
#    df_csmp_prms.to_csv(f'{out_dir_ini}/csmp_prms_N{N}_p{p}.csv')
#    S_rng = list(range(5,int(N/5+1)))
#    S_opt,c_optcs,mn_vlerr_cscv,mn_trnerr_cscv= omu.cross_valid_cosamp(Psi_csmp,u_data[optim_indices],S_rng,csmp_prms,n_fold=5)
#    trnerr_p_optcs, vlderr_p_optcs = tnn.val_test_err(data_tst,mi_mat,c_optcs)
#    df_ccs_opt = pd.DataFrame({'c_cs':c_optcs})
#    df_ccs_opt.to_csv(f'{out_dir_ini}/plots/j={j}/c_optcs_1dellps_n={N}_genmod_S={S_opt}_p={p}_j{j}.csv',index=False)
#    df_epsccs_opt = pd.DataFrame({'vlderr_p_cs':vlderr_p_optcs,'trnerr_p_cs':trnerr_p_optcs},index=[0])
#    df_epsccs_opt.to_csv(f'{out_dir_ini}/plots/epsu_csph_tst_1dellps_n={N}_genmod_S={S_opt}_j{j}.csv',index=False)
#    plt.figure(4+j)
#    fig, ax = plt.subplots()
#    txtstrng = '\n'.join([r'$S_{opt}=Sop$'.replace('Sop',str(S_opt)),'R(k) = $||\mathbf{\Psi}(k) \mathbf{c} - \mathbf{u}(k)||_2$'])
#    ax.plot(S_rng,mn_vlerr_cscv,'--*r',label='valid')
#    ax.plot(S_rng,mn_trnerr_cscv,'--ob',label='train')
#    ax.set_xlabel('Sparsity (S)')
#    ax.set_ylabel(r'$\sum_{k=1}^{n_f} R(k) / n_f $')
#    ax.text(0.35,0.85,txtstrng,transform=ax.transAxes,bbox=dict(facecolor='white',edgecolor='black'))
#    ax.legend()
#    ax.grid()
#    plt.tight_layout()
#    plt.savefig(f'{out_dir_ini}/csmp_cv_errplt_j{j+1}.png')
#import pdb; pdb.set_trace()    
#=================================================================================
#for testing purposes: test the sparsity values for different sample replications 
#=================================================================================
S_rep_test = []; S_rep_test_ph = []
for j in j_rng:
    print(f'=============#replication={j}============')
    ecmn_ind = np.zeros(tot_itr)
    os.makedirs(f'{out_dir_ini}/plots/j={j}',exist_ok=True)
    optim_indices = indices0.iloc[j].to_numpy()
    valid_indices = np.setdiff1d(range(np.size(u_data)), optim_indices)
    trains = [name for name in indices0.columns if name.startswith("optim")]
    test_indices = indices0.loc[j][trains].to_numpy()
#    import pdb; pdb.set_trace()
    data_tst = {'y_data':y_data,'u_data':u_data,'val_ind':valid_indices,'test_ind':test_indices,'opt_ind':optim_indices,'Nv':Nv,
            'chc_poly':chc_Psi,'chc_omp':chc_omp_slv} 
    c_ini, S_omp0, train_err_p0, valid_err_p0,P_omp,mi_mat_omp, Psi_omp = omu.omp_utils_order_ph(out_dir_ini,d,p_0,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_omp0,j)
    c_omph, S_omph, test_omp_ph, valid_omp_ph,P_omph,mi_mat_omph, Psi_omph= omu.omp_utils_order_ph(out_dir_ini,d,p,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_omp,j)
    S_rep_test.append(S_omp0)
    S_rep_test_ph.append(S_omph)
S_rep_avg = sum(S_rep_test)/len(S_rep_test)    
df_S_reptest = pd.DataFrame({'S_lst':np.array(S_rep_test)})
df_S_reptest.to_csv(f'{out_dir_ini}/plots/S_rep_list.csv',index=False)
df_S_reptest_ph = pd.DataFrame({'S_lst':np.array(S_rep_test_ph)})
df_S_reptest_ph.to_csv(f'{out_dir_ini}/plots/S_rep_list_ph.csv',index=False)
#=================================================================================
#=================================================================================
import pdb; pdb.set_trace()
for j in j_rng:
    print(f'=============#replication={j}============')
    ecmn_ind = np.zeros(tot_itr)
    os.makedirs(f'{out_dir_ini}/plots/j={j}',exist_ok=True)
    optim_indices = indices0.iloc[j].to_numpy()
    valid_indices = np.setdiff1d(range(np.size(u_data)), optim_indices)
    trains = [name for name in indices0.columns if name.startswith("optim")]
    test_indices = indices0.loc[j][trains].to_numpy()
#    import pdb; pdb.set_trace()
    data_tst = {'y_data':y_data,'u_data':u_data,'val_ind':valid_indices,'test_ind':test_indices,'opt_ind':optim_indices,'Nv':Nv,
            'chc_poly':chc_Psi,'chc_omp':chc_omp_slv} 
#======================================================================================
#=====================Testing effect of sparsity on omp===============================
#======================================================================================
    #for S_tst in range(1800,2100,100):
    #    c_ini, S_omp0, train_err_p0, valid_err_p0,P_omp,mi_mat_omp, Psi_omp = omu.omp_utils_order_ph(out_dir_ini,d,p_0,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_omp0,j)
#    import pdb; pdb.set_trace()
#======================================================================================
#======================================================================================
#======================================================================================
##   Calculate validation error using specific coeff: top-x values are kept as nonzero
    #c_test_fl = pd.read_csv('../output/titan_ppr/ompcv/ini/rnd_ceff/comp_fl_1dellps_n=100_genmod_S=6_0_j0_c1.csv').to_numpy().flatten()
#    c_test_fl = pd.read_csv('../output/titan_ppr/results/csaug13/d=21/p=5/ompcv/N=7k/plots/j=0/comp0_1dellps_n=7000_genmod_S=166_p=5_j0.csv').to_numpy().flatten()
#    valid_err_tst = 100.0; tst_ind = 0
#    c_test = np.zeros(P)
##    while valid_err_tst > 0.02:
#    for tst_ind in range(70,110,10):
#        c_P_test = np.copy(c_test_fl)
#        c_test[:] = c_P_test[:P]
#        P_test = np.size(c_test)
#        c_srt_ind = np.argsort(np.abs(c_test))[::-1]
#    #    import pdb; pdb.set_trace()
#        bot_7_ind = np.setdiff1d(list(range(P_test)),c_srt_ind[:7+tst_ind])
#        c_test[bot_7_ind] = 0.0
#        S_test = np.size(np.nonzero(c_test)[0])
#        print("nonzero c_test:",np.nonzero(c_test))
#    #    c_test[13] = 0.0; #c_test[69] = 0.0;
#    #   c_test[2] = 0.0; #c_test[13] = 0.0;
#    #    c_test[229] = 0.0; c_test[2] = 0.0;
#         
#        train_err_tst, valid_err_tst = tnn.val_test_err(data_tst,mi_mat,c_test)
#        print('valid_err_tst',valid_err_tst)
#        df_ctest = pd.DataFrame({'c_test':c_test})
#        df_ctest.to_csv(f"{out_dir_ini}/c_test_S_test{S_test}tst_ind{tst_ind}.csv")  
#        df_errtest = pd.DataFrame({'vlerr_test':valid_err_tst,'trnerr_test':train_err_tst},index=[0])
#        df_errtest.to_csv(f"{out_dir_ini}/err_test_S_test{S_test}_tst_ind{tst_ind}.csv")  
#        #tst_ind +=1
#    import pdb; pdb.set_trace()
#======================================================================================
#======================================================================================
#======================================================================================
# Take the chat coefficients and calculate \epsilon_u for specific coefficient:#TODO:comment!!!
    #c_test_fl = pd.read_csv('../output/titan_ppr/results/csjul29_rsdl/plots/j=17/it=4/c_hat_tot_1dellps_n=100_genmod_S=6_0_j17_c0.csv')['c_hat'].to_numpy().flatten()
#    c_test_fl = pd.read_csv('../output/titan_ppr/results/csjul29_rsdl/plots/j=17/it=4/comp_fl_1dellps_n=100_genmod_S=6_4_j17_c0.csv')['comp_fnl'].to_numpy().flatten()
#    c_test = np.copy(c_test_fl)
#    c_test[3145] = 0.0
#    c_test[74] = -3e-4
#    train_err_tst, valid_err_tst = tnn.val_test_err(data_tst,mi_mat,c_test)
#    import pdb;pdb.set_trace()
#%% write Psi into a csv file:
    # Psi = pcu.make_Psi(y_data,mi_mat)     #remember Psi should be factored by sqrt(2)
    # df_Psi = pd.DataFrame(Psi)
    # df_Psi.to_csv(f'{out_dir_ini}/Psi_pd={p,d}.csv',index=False)
    
    ## read Psi from a csv file:
    # Psi = pd.read_csv(f'{out_dir_ini}/Psi_pd={p,d}.csv').to_numpy()
    # ref_ind = random.sample(range(40000), 10000)
    # u_ref = Psi[ref_ind,:] @ c_ref
    # u_diff = u_data[ref_ind] - u_ref
    # eps_u = la.norm(u_diff)/la.norm(u_data[ref_ind])
    # #Calculate valid/test error:
    # c_omd = pd.read_csv(f'{out_dir_ini}/ini/ompcs/comp_fl_1dellps_n=680_genmod_S=168_0.csv').to_numpy().flatten()
    # N_tep = 1000 
    # eps_u_500 = la.norm(Psi[valid_indices[:500],:] @ c_ref - u_data[valid_indices[:500]])/la.norm(u_data[valid_indices[:500]])
    # eps_u_t = la.norm(Psi[test_indices[:N_tep],:] @ c_omd - u_data[test_indices[:N_tep]])/la.norm(u_data[test_indices[:N_tep]])
    # Psi_N = Psi[optim_indices,:]
    #Least square solution:
    # Psi_T = np.transpose(Psi)
    # c_ls = (la.inv(Psi_T @ Psi) @ Psi_T @ u_data).flatten()    
#    %% Least square solution:#FIXME:
#    c_ls = np.zeros(P)
#    Lam_ls =  [ 0,60,2,1,74,9,51,3145] #top-6 true coefficients in descending order: [ 0, 60, 2,1,74,9]
#    Psi = pcu.make_Psi_drn(y_data[optim_indices,:],mi_mat,Lam_ls,chc_Psi)     
#    Psi_T = np.transpose(Psi)
#    c_ls_sht = (la.inv(Psi_T @ Psi) @ Psi_T @ u_data[optim_indices]).flatten()
#    c_ls[Lam_ls] = c_ls_sht
#    print('c_ls',c_ls)
#    trn_err_ls, valid_err_ls = tnn.val_test_err(data_tst,mi_mat,c_ls)
#test the validation error of another coefficient:

#======================================================================================
#============Apply least squares for the particular active set=========================
#======================================================================================
    #Lam_ls =[0,21,2,4,3,82,16,99] #top-10: [ 0 21  2  4  3 82 16 20 18 99]
    #c_ls = np.zeros(P)
    #Psi = pcu.make_Psi_drn(y_data[optim_indices,:],mi_mat,Lam_ls,chc_Psi)     
    #Psi_T = np.transpose(Psi)
    #c_ls_sht = (la.inv(Psi_T @ Psi) @ Psi_T @ u_data[optim_indices]).flatten()
    #c_ls[Lam_ls] = c_ls_sht
    #print('c_ls',c_ls)
    #trn_err_ls, valid_err_ls = tnn.val_test_err(data_tst,mi_mat,c_ls)
    #import pdb; pdb.set_trace()
#======================================================================================
#======================================================================================
#======================================================================================
#======================================================================================
    #%% Take two basis at a time and apply least squares:
    #    Lam_ls = pd.read_csv(f'{out_dir_ini}/results/csjul24_rsdl/plots/j=0/it=1/Lam_sel_1dellps_n=100_genmod_S=6_1_j0_c0.csv')['Lam_sel'].to_numpy().flatten() 
    #    Lam_ls = np.setdiff1d(Lam_ls,3145)
#    Lam_lsfl = pd.read_csv(f'{out_dir_ini}/results/csjul24_rsdl/plots/j=0/it=1/Lam_sel_1dellps_n=100_genmod_S=6_1_j0_c0.csv')['Lam_sel'].to_numpy().flatten()
#    c_fl_ls = np.zeros((P,np.size(Lam_lsfl) - 1))                               
#    vler_fl_ls = []
#    trer_fl_ls = []
#    for ls_ind in range(np.size(Lam_lsfl)-1):
#        Lam_ls = [Lam_lsfl[0],Lam_lsfl[ls_ind+1]]
#        c_ls = np.zeros(P)
#    ##    Psi_fl = pd.read_csv(f'{out_dir_ini}/plots/j={j}/Psi_omph_1dellps_n={N}_genmod_p={p}_d{d}_j{j}.csv').to_numpy()   
#        Psi = pcu.make_Psi_drn(y_data[optim_indices,:],mi_mat,Lam_ls,chc_Psi)     
#        Psi_T = np.transpose(Psi)
#        c_ls_sht = (la.inv(Psi_T @ Psi) @ Psi_T @ u_data[optim_indices]).flatten()
#        c_ls[Lam_ls] = c_ls_sht
#        c_fl_ls[:,ls_ind] = c_ls
#        print('c_ls',c_ls)
#        trn_err_ls, valid_err_ls = tnn.val_test_err(data_tst,mi_mat,c_ls)
#        vler_fl_ls.append(valid_err_ls)
#        trer_fl_ls.append(trn_err_ls)
##    res_ls = Psi_fl[:,Lam_ls] @ c_ls[Lam_ls] - u_data[optim_indices]
#    import pdb; pdb.set_trace()
##    Psi_fl = pcu.make_Psi(y_data[optim_indices,:],mi_mat,chc_Psi)
#    lam_opt = np.argmax(np.abs(Psi_fl.T @ res_ls))
#    import pdb; pdb.set_trace()
   #  df_cls = pd.DataFrame({'c_ls':c_ls})
   #  df_cls.to_csv(f'{out_dir_ini}/plots/cls_1dellps_n={N}_genmod_p={p}_j{j}.csv',index=False)
   #  df_epsuls = pd.DataFrame({'epsu_ls':valid_err_ls,'epsu_ls_tr':trn_err_ls},index=[0])
   #  df_epsuls.to_csv(f'{out_dir_ini}/plots/epsuls_tst_1dellps_n={N}_p={p}_genmod_j{j}.csv',index=False)
   #  import pdb; pdb.set_trace()
   # import pdb; pdb.set_trace()   
#======================================================================================
#===============================Apply CoSaMP without CV================================
#======================================================================================
#    Psi_csmp = pcu.make_Psi(y_data[optim_indices,:d],mi_mat,chc_Psi)
#    S_cs = 7; maxit_cs = 10
#    c_cs, resnrm_cs = omu.cosamp_func(Psi_csmp,u_data[optim_indices],S_cs,max_iter=maxit_cs,hlt_crit='iter',tol_res=1e-4)
#    df_ccs = pd.DataFrame({'c_cs':c_cs})
#    df_ccs.to_csv(f'{out_dir_ini}/plots/j={j}/ccs_1dellps_n={N}_genmod_S={S_cs}_p={p}_j{j}_it{maxit_cs}.csv',index=False)
#    df_rsnrmcs = pd.DataFrame({'rsrnm':np.array(resnrm_cs)})
#    df_rsnrmcs.to_csv(f'{out_dir_ini}/plots/j={j}/rsnrmcs_1dellps_n={N}_genmod_S={S_cs}_p={p}_j{j}_it{maxit_cs}.csv',index=False)
#    trnerr_p_cs, vlderr_p_cs = tnn.val_test_err(data_tst,mi_mat,c_cs)
#    df_epsccs = pd.DataFrame({'vlderr_p_cs':vlderr_p_cs,'trnerr_p_cs':trnerr_p_cs},index=[0])
#    df_epsccs.to_csv(f'{out_dir_ini}/plots/epsu_csph_tst_1dellps_n={N}_genmod_S={S_cs}_j{j}_it{maxit_cs}.csv',index=False)
#    plt.figure(3)
#    plt.plot(resnrm_cs,'--*r',label='CoSaMP')
#    plt.xlabel('iteration index (k)')
#    plt.ylabel(r'$||\mathbf{\Psi c} - \mathbf{u}||_2$')
#    plt.legend()
#    plt.grid()
#    plt.savefig(f'{out_dir_ini}/res_norm_cs.png')
#    import pdb; pdb.set_trace()
#======================================================================================
#===============CV procedure to find S_opt using CoSaMP================================
#======================================================================================
#    Psi_csmp = pcu.make_Psi(y_data[optim_indices,:d],mi_mat,chc_Psi)
#    csmp_prms = {'maxit_csmp':10,'hlcrt_csmp':'iter','tolres_csmp':1e-2,'rnd_st_cvcs': 1} #'iter'
#    df_csmp_prms = pd.DataFrame(csmp_prms,index=[0])
#    df_csmp_prms.to_csv(f'{out_dir_ini}/csmp_prms_N{N}_p{p}.csv')
#    S_rng = list(range(5,int(N/5+1)))
#    S_opt,c_optcs,mn_vlerr_cscv,mn_trnerr_cscv= omu.cross_valid_cosamp(Psi_csmp,u_data[optim_indices],S_rng,csmp_prms,n_fold=5)
#    trnerr_p_optcs, vlderr_p_optcs = tnn.val_test_err(data_tst,mi_mat,c_optcs)
#    df_ccs_opt = pd.DataFrame({'c_cs':c_optcs})
#    df_ccs_opt.to_csv(f'{out_dir_ini}/plots/j={j}/c_optcs_1dellps_n={N}_genmod_S={S_opt}_p={p}_j{j}.csv',index=False)
#    df_epsccs_opt = pd.DataFrame({'vlderr_p_cs':vlderr_p_optcs,'trnerr_p_cs':trnerr_p_optcs},index=[0])
#    df_epsccs_opt.to_csv(f'{out_dir_ini}/plots/epsu_csph_tst_1dellps_n={N}_genmod_S={S_opt}_j{j}.csv',index=False)
#    plt.figure(4)
#    fig, ax = plt.subplots()
#    txtstrng = '\n'.join([r'$S_{opt}=Sop$'.replace('Sop',str(S_opt)),'R(k) = $||\mathbf{\Psi}(k) \mathbf{c} - \mathbf{u}(k)||_2$'])
#    ax.plot(S_rng,mn_vlerr_cscv,'--*r',label='valid')
#    ax.plot(S_rng,mn_trnerr_cscv,'--ob',label='train')
#    ax.set_xlabel('Sparsity (S)')
#    ax.set_ylabel(r'$\sum_{k=1}^{n_f} R(k) / n_f $')
#    ax.text(0.35,0.85,txtstrng,transform=ax.transAxes,bbox=dict(facecolor='white',edgecolor='black'))
#    ax.legend()
#    ax.grid()
#    plt.tight_layout()
#    plt.savefig(f'{out_dir_ini}/csmp_cv_errplt.png')
#
#    import pdb; pdb.set_trace()
    #%% OMP coefficients:
    # Find initial signs with orthogonal matching pursuit (sklearn):
##==========================================================================================================
#======================================================================================
#======================================================================================
    #import pdb; pdb.set_trace()
    if args.debug_alg==1:
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
    c_omph, S_omph, test_omp_ph, valid_omp_ph,P_omph,mi_mat_omph, Psi_omph= omu.omp_utils_order_ph(out_dir_ini,d,p,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_omp,j)
#=============================================================================
    # #p=3
    print('S_omph:',S_omph)
    print(f'Training Error for c_omph: {test_omp_ph}')
    # Validation error: is the error on the unseen data:
    print(f'Validation Error for c_omph: {valid_omp_ph}')
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
    # plt.figure(200)
    # plt.semilogy(np.abs(c_ini),'g.',label='$c_{omph}$', markersize=10)
    # plt.semilogy(np.abs(c_ref),'r*',label='$c_{fl}$')
    # # plt.xlim([0,120])
    # plt.legend()
    # # plt.ylabel('')
    # plt.show()
    # OMP using full function:
    # mi_mat_omp = pcu.make_mi_mat(d_omp, p_omp)
    # P_omp = np.size(mi_mat_omp,0)   i   import pdb;pdb.set_trace()

    # Psi_omp = pcu.make_Psi(y_data[optim_indices,:],mi_mat_omp,chc_Psi)
    # # OMP solutions:
    # c_om_std1,S_fnl = omp1.main_omp(P_omp,sprsty1,N,Psi_omp,u_data[optim_indices].flatten())  
    # print('S_size:',len(S_fnl))  
    # c_ini = c_om_std1
    # plt.figure(222)
    # plt.semilogy(np.abs(c_omph),'g.', markersize=10)
    # plt.semilogy(np.abs(c_ini),'r*')
    # # #%%calculate sparsity:
    # sprsty = np.size(np.nonzero(c_ini)[0])

    #%% Plot active sets:
    # df_cini_omp = pd.DataFrame({'cini':c_ini})
    # df_cini_omp.to_csv(f'{out_dir_ini}/plots/cini_1dellps_n={N}_genmod_S={sprsty}_p={p_omp}.csv',index=False)
#    import pdb; pdb.set_trace()
    plt.figure(110)
    max_nnzr = np.size(np.nonzero(c_ini))
    plt.plot(np.nonzero(c_ini),np.nonzero(c_ini),'*r')
    plt.xlabel('Index of PCE coefficients')
    plt.ylabel('Active sets')  
    #%% initial parameters:
    # # sprsty = 43
    # tot_itr = 3
    # # sprsty = 5  #Think about giving sparsity=1, some matrix manipulations might get affected.
    # learning_rate = 0.001
    # epochs = 10**1
    multi_ind_mtrx = torch.Tensor(mi_mat)
#    print('GNNmod:',GNNmod)
#    import pdb;pdb.set_trace()
    # nn.utils.vector_to_parameters(thet_str, GNNmod.parameters())
    # nn.utils.vector_to_parameters(torch.Tensor(thet_str), GNNmod.parameters())
    # G_str = GNNmod(multi_ind_mtrx).detach().numpy().flatten()
    # thet_str = pd.read_csv(f'{out_dir_ini}/ini/mxit1k/1dellps_n=100_genmod_prms_0.csv').to_numpy().flatten()
    # thet_str = pd.read_csv(f'{out_dir_ini}/ini/thet_up/NNexp/zprms_1dellps_n=300_genmodNN_exp_p=3_S=20.csv').to_numpy().flatten()
    # thet_str = thet_str1['thet_upd'].to_numpy().flatten()
    # thet_str = thet_str + 0.01* np.random.rand(z_n)
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
    opt_params = {'ph':p,'p0':p_0,'d':d,'epochs':epochs,'lr':learning_rate,'S':sprsty,'S0':S_omp0,
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
    # lm_cht0_so = [ 0,  1,  2,  3, 17, 18, 19, 34, 39, 48]
    # lm_cht0_mo = [  0,   1,   2,   3,   4,   8,  11,  13,  15,  16,  17,  18,  19,
    #      33,  34,  39,  48, 169, 289, 290] 
    # mi_mat_ls = mi_mat_omph
    # P_ls = P_omph
    # Lam_ls = lm_cht0_mo
    # Psi_active_t = pcu.make_Psi_drn(y_data[optim_indices,:d],mi_mat_ls,Lam_ls,chc_Psi)
    # Psi_active_t_T = np.transpose(Psi_active_t)
    # c_ls_t = (la.inv(Psi_active_t_T @ Psi_active_t) @ Psi_active_t_T @ u_data[optim_indices]).flatten()
    # c_ls_t_full = np.zeros(P_ls)
    # c_ls_t_full[np.array(Lam_ls)] = c_ls_t    
    #%% Define the training function:
    #initialization:
    #nn.utils.vector_to_parameters(thet_upd, GNNmod.parameters())
    #print("GNNmod parameters:",GNNmod.parameters())
    #Train on a few points:
    # mi_mat_in = mi_mat_omp[0:2,:]
    # c_hat = c_ini[0:2]    
    # mi_mat_in = mi_mat_omp[0,:]
    # c_hat = c_ini[0]
    mi_mat_in = mi_mat_omp
    c_hat = np.copy(c_ini)
    cr_mxind = (np.argsort(np.abs(c_hat))[::-1])[:top_i0]
    # mi_mat_in = mi_mat_omp[S_fnl,:]
    # c_hat = c_ini[S_fnl]
#    test_err, valid_err = tnn.val_test_err(data_tst,mi_mat_omp,c_ini)
    eps_u[0] = valid_err_p0
    
#    # Testing error: is the error on the training data:
#    print(f'Testing Error: {test_err}')
#    test_err_ls.append(test_err) 
#    # Validation error: is the error on the unseen data:
#    print(f'Validation Error: {valid_err}')   
#    valid_err_ls.append(valid_err)
    #%% Verify NN:
    # G_ver = dmold.NN_Gapprx(thet_str, mi_mat_in,Hid) 
    # print('G_ver',G_ver)
    #%% Start training:
    ls_vlit_min = np.zeros((Nrp_vl,tot_itr))
    #define the config dictionary for tuning with Ray Tune:
    #config_tune = {'Hid':tune.sample_from(lambda _:np.random.randint(2,3)),'lr':learning_rate}
    #layers_cnfig = [d] + [tune.randint(7,7) for __ in range(Nlhid)] +[1]
    #layers_cnfig = [d,Hid,1]
    config_tune = {'lr':tune.choice([learning_rate])}
    for layer in range(len(hid_layers)): 
        config_tune[f'h{layer}'] = hid_layers[layer]  
        config_tune[f'a{layer}'] = avtnlst[layer] 
    #import pdb; pdb.set_trace()    
    #initalize i:    
    i = 0
    while i < tot_itr:
        print(f'=============total iteration index={i}============')
        os.makedirs(f'{out_dir_ini}/plots/j={j}/it={i}',exist_ok=True)
        # print('G_mod:',G_mod) 
        # optimizer = torch.optim.SGD(GNNmod.parameters(), lr=learning_rate)        
        #optimizer = torch.optim.Adam(GNNmod.parameters(), lr=learning_rate)
        #cost_val_tmp = np.zeros((int(epochs/freq),Nrp_vl)) 
        #cost_tmp = np.zeros_like(cost_val_tmt_
        #thet_f_tmp = np.zeros((z_n,Nrp_vl))
        #cost_val_tmp = {}
        #cost_tmp = {}
        #thet_bst_tmp = {}
        #thet_f_tmp = torch.zeros((z_n,Nrp_vl))
        #thet_bst_tmp = torch.zeros((z_n,Nrp_vl))
        #zer_str_tmp = np.zeros_like(cost_tmp)
        for trc in range(Nc_rp):
            if i>0:
                #mi_mat_in = mi_mat   
                #uncomment to run the code with the corresponding validation coefficients used for the previous iteration:
                # c_hat = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/comp_fl_1dellps_n={N}_genmod_S={sprsty}_{i-1}_j{j}_c{trc}.csv').to_numpy().flatten()   
                #c_hat = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/comp_fl_1dellps_n={N}_genmod_S={sprsty}_{i-1}_j{j}_c{int(ecmn_ind[i-1])}.csv').to_numpy().flatten()
                # NOTe: Here you won't be using more than Nc_rp=1.
                c_hat = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/comp_rs_1dellps_n={N}_genmod_S={sprsty}_{i-1}_j{j}_c{trc}.csv').to_numpy().flatten()
                
                cr_mxind = (np.argsort(np.abs(c_hat))[::-1])[:top_i1]

            P_alg = np.size(mi_mat_in,0)
            cini_nz = np.nonzero(c_hat)[0]
            cini_nz_ln = np.size(cini_nz) 
            cini_z = np.setdiff1d(range(P_alg),cini_nz)
            cini_z_ln = np.size(cini_z)
            print('cini_nz',cini_nz)
            #get training indices such that 4-5ths of nonzeros are retained:
            # trn_ind_nz = random.sample(cini_nz.tolist(), int(4*cini_nz_ln/5)) 
            # trn_ind_z = random.sample(cini_z.tolist(),int(4*cini_z_ln/5))
            # trn_ind_nw = np.concatenate((trn_ind_nz,trn_ind_z),axis=None)
            #to include/exclude top 10 coefficients in the validation/training sets:
            cini_sbmind = np.setdiff1d(cini_nz,cr_mxind)
            print('cini_sbmind',cini_sbmind)
            print('cr_mxind',cr_mxind)
            ntpk_cr = np.size(cr_mxind)
            #import pdb;pdb.set_trace()
            random.seed(seed_ceff+j)
            rnd_smp_dict = {'cr_mxind':cr_mxind,'cini_sbmind':cini_sbmind,'cini_nz_ln':cini_nz_ln,'cini_z':cini_z,'ntpk_cr':ntpk_cr,'cini_z_ln':cini_z_ln,'cini_nz':cini_nz}
            #for itvl_ind in range(Nrp_vl):
            trn_ind_nz = random.sample(cini_sbmind.tolist(), int(4*cini_nz_ln/5-ntpk_cr)) #to include all top 10.
            #print('trn_ind_nz',trn_ind_nz)
            ##trn_ind_nz = random.sample(cini_sbmind.tolist(), int(4*cini_nz_ln/5)) 
            trn_ind_z = random.sample(cini_z.tolist(),int(4*cini_z_ln/5))
            #added lines to be uncommented
            #=============================================================
            if args.debug_alg==0: 
                config_tune['tind_nz'] = tune.sample_from(lambda _: random.sample(cini_sbmind.tolist(), int(4*cini_nz_ln/5-ntpk_cr-args.vlcfadd)))

                config_tune['tind_z'] =  tune.sample_from(lambda _: random.sample(cini_z.tolist(),int(4*cini_z_ln/5+args.vlcfadd)))
            elif args.debug_alg==2:
                df_trn_ind = pd.read_csv('/home/jothi/CoSaMP_genNN/output/titan_ppr/results/csaug13/d=21/p=3/ref_dbg/plots_abs/trn_indices_alph_omp_N=100_0_c0.csv')
                trn_ind_dbg_fl = df_trn_ind['trn_ind_nw'].to_numpy() 
                t_indnz_sbmx_dbg = [3,4,16,196]
                trn_ind_dbg_1 = np.setdiff1d(trn_ind_dbg_fl,cr_mxind)
                trn_indz_dbg = np.setdiff1d(trn_ind_dbg_1,t_indnz_sbmx_dbg) 
                config_tune['tind_nz'] = tune.sample_from(lambda _:t_indnz_sbmx_dbg)
                config_tune['tind_z'] =  tune.sample_from(lambda _:trn_indz_dbg) 
                #import pdb; pdb.set_trace()
            print('config_tune',config_tune)
            #=============================================================
            # trn_ind_nw = np.concatenate((trn_ind_nz,trn_ind_z),axis=None) 
            #trn_ind_nw = np.concatenate((cr_mxind,trn_ind_nz,trn_ind_z),axis=None)
            #Test indices: 
            # trn_ind_nw = random.sample(range(P_alg), int(4*P_alg/5))
    
            #df_trn_ind_nw = pd.DataFrame({'trn_ind_nw':trn_ind_nw})        
            #df_trn_ind_nw.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/trn_indices_alph_omp_N={N}_{i}_c{trc}.csv',index=False)
            ## Validation indices:    
            #val_ind_nw = np.setdiff1d(np.linspace(0,P_alg-1,P_alg),trn_ind_nw)
            #df_val_ind_nw = pd.DataFrame({'val_ind_nw':val_ind_nw})        
            #df_val_ind_nw.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/val_indices_alph_omp_N={N}_{i}_c{trc}.csv',index=False)
            #==========================================================================
            #read val,train indices:
            #if i==0:               
            #    # FIXME: HARDCODING:
            #    trn_ind_nw = pd.read_csv(f'/home/jothi/CoSaMP_genNN/output/titan_ppr/results/csaug13/d=21/p=3/ref_data/trn_indices_alph_omp_N=100_0_c0_ivl0.csv').to_numpy().flatten()
            #    val_ind_nw = pd.read_csv(f'/home/jothi/CoSaMP_genNN/output/titan_ppr/results/csaug13/d=21/p=3/ref_data/val_indices_alph_omp_N=100_0_c0_ivl0.csv').to_numpy().flatten()
            # elif i==1:  
            #     trn_ind_nw = pd.read_csv(f'{out_dir_ini}/ini/ind/N=100/trn_indices_alph_omp_N=100_1_c0.csv').to_numpy().flatten()
            #     val_ind_nw = pd.read_csv(f'{out_dir_ini}/ini/ind/N=100/val_indices_alph_omp_N=100_1_c0.csv').to_numpy().flatten()
            # else:
            #     trn_ind_nw = pd.read_csv(f'{out_dir_ini}/ini/ind/epsc_dec/trn_indices_alph_omp_N=200_1_c0.csv').to_numpy().flatten()
            #     val_ind_nw = pd.read_csv(f'{out_dir_ini}/ini/ind/epsc_dec/val_indices_alph_omp_N=200_1_c0.csv').to_numpy().flatten()
            #df_trn_ind_nw = pd.DataFrame({'trn_ind_nw':trn_ind_nw})        
            #df_trn_ind_nw.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/trn_indices_alph_omp_N={N}_{i}_c{trc}_ivl{itvl_ind}.csv',index=False)
            #df_val_ind_nw = pd.DataFrame({'val_ind_nw':val_ind_nw})        
            #df_val_ind_nw.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/val_indices_alph_omp_N={N}_{i}_c{trc}_ivl{itvl_ind}.csv',index=False)
            #========================================================================
            #Ray tune Tuning:
            #========================================================================
           # scheduler = ASHAScheduler(metric='loss_met', mode='min', max_t=epochs,grace_period=1,reduction_factor=2) 
            scheduler = AsyncHyperBandScheduler(max_t=epochs,grace_period=1,reduction_factor=2) 
            print(":after scheduler:")
            #reporter = CLIReporter(metric_columns=["loss_met",'training iteration']) 
            print(":after reporter:")
           # import pdb;pdb.set_trace()
            #FIXME: the training stops once the validation error starts increasing: sometimes, you might want to use more epochs. 
            part_fnc = partial(tnn.train_theta,torch.Tensor(np.abs(c_hat.flatten())), thet_upd,thet_str,i, 
                torch.Tensor(mi_mat_in),epochs,freq,W_fac[i],avtnlst,Nlhid,tune_sg,args.it_fix,rnd_smp_dict)
            print(":after partfunc:")
            tuner = tune.Tuner(part_fnc,run_config=air.RunConfig(storage_path=f'{out_dir_ini}') ,tune_config=tune.TuneConfig(metric='loss_met', mode='min',scheduler=scheduler,num_samples=num_trial), param_space=config_tune)
            #import pdb;pdb.set_trace()
            print(":after results:")
            result = tuner.fit() 
            #result_df = result.get_dataframe() 
            best_result = result.get_best_result()
            best_config = best_result.config
            with open(f'{out_dir_ini}/plots/j={j}/it={i}/best_config.pickle','wb') as bprms_pickl:
                pickle.dump(best_config,bprms_pickl)
            print("Best hyperparameters found were: ",best_config)
            GNNmod = gnn.GenNN([d] + [best_config.get(f'h{lyr}') for lyr in range(Nlhid)] +[1])
            #import pdb; pdb.set_trace()
            best_result_df = best_result.metrics_dataframe
            best_epoch = best_result_df['ep_best'].to_numpy().flatten() 
            #cost = best_result_df['train_loss'].to_numpy().flatten() 
            #retrieve the best model:
            best_checkpoint = best_result.checkpoint  
            best_chckpnt_dict = best_checkpoint.to_dict() 
            thet_bst = best_chckpnt_dict.get("thet")      
            cost = best_chckpnt_dict.get("train_app")      
            cost_val = best_chckpnt_dict.get("val_app")      
            torch.save(best_chckpnt_dict,f'{out_dir_ini}/plots/j={j}/it={i}/model_best_cpt_i{i}_j{j}.pt')

            #retrives initial checkpoint:
            bcpt_path = best_checkpoint.path
            index_cpt = bcpt_path.find("checkpoint_")
            index_uscr = index_cpt+len("checkpoint_")
            add_strng_pth =  str(0).zfill(len(bcpt_path[index_uscr:]))
            ini_path = bcpt_path[:index_uscr] +add_strng_pth
            inl_cptdir  = Checkpoint.from_directory(ini_path)
            inl_cpt_dict = inl_cptdir.to_dict()
            torch.save(inl_cpt_dict,f'{out_dir_ini}/plots/j={j}/it={i}/model_ini_cpt_i{i}_j{j}.pt')
            #import pdb; pdb.set_trace()
            #retrives final checkpoint:
            add_strng_pth_fnl =  str(1).zfill(len(bcpt_path[index_uscr:]))
            final_path = bcpt_path[:index_uscr] +add_strng_pth_fnl
            final_cptdir  = Checkpoint.from_directory(final_path)
            final_cpt_dict = final_cptdir.to_dict()
            torch.save(final_cpt_dict,f'{out_dir_ini}/plots/j={j}/it={i}/model_fnl_cpt_i{i}_j{j}.pt')
            #import pdb; pdb.set_trace()
            #FIXME:seeding for initializing theta in the next iteration
            np.random.seed(i+sd_thtini_2nd)
            thet_upd = torch.Tensor(np.random.rand(z_n))
# Write temp    orary variables:
            ## Least squares step at low validation error:          
            c_omp_bst = np.zeros(P)
            c_omp_sel = np.zeros(P)
#====================NOTe HERE thet_upd was there in the place of thet_bst=========            
            nn.utils.vector_to_parameters(torch.Tensor(thet_bst), GNNmod.parameters())
            torch.save(GNNmod,f'{out_dir_ini}/plots/j={j}/it={i}/model_final_i{i}_j{j}.pt')
            torch.save(GNNmod.state_dict(),f'{out_dir_ini}/plots/j={j}/it={i}/modelprms_final_dict_i{i}_j{j}.pt')
            Gmod_bst = GNNmod(torch.Tensor(mi_mat),[best_config.get(f'a{lyr1}') for lyr1 in range(Nlhid)],i).detach().numpy().flatten()
#=================For randomly initializing \theta (I think it is redundant)=================================            
#            nn.utils.vector_to_parameters(thet_upd, GNNmod.parameters())           
#Test the idea of picking many basis functions and keeping only the most important:
           # Psi_test = pcu.make_Psi(y_data[optim_indices,:d],mi_mat,chc_Psi)
#                import pdb; pdb.set_trace()
#                Lambda_bst = (np.argsort(Gmod_bst)[::-1])[:sprsty]
#                Psi_test = pcu.make_Psi(y_data[optim_indices,:d],mi_mat,chc_Psi)
#%% approximate residual signal:
            
            if i==0:
                Lambda_sel_tmp = (np.argsort(Gmod_bst)[::-1])[:S_chs]               
                Lambda_sel = Lambda_sel_tmp
            else:
                Lambda_sel_tmp = (np.argsort(Gmod_bst)[::-1])[:S_chs]
                Lam_pr_bst  = pd.read_csv(f'{out_dir_ini}/plots/j={j}/it={i-1}/Lam_bst_1dellps_n={N}_genmod_S={sprsty}_{i-1}_j{j}_c{trc}.csv')['Lam_bst'].to_numpy().flatten()
                Lam_comn = np.intersect1d(Lambda_sel_tmp,Lam_pr_bst)
                S_comn = Lam_comn.size
                S_csit = sprsty+S_chs-S_comn
                print("Lambda_sel_tmp",Lambda_sel_tmp)
                print("Lambda_pr_bst",Lam_pr_bst)
                print("Lam_comn",Lam_comn)

                if S_csit > S_chs:
                    Lambda_sel_tmp = Lambda_sel_tmp[np.in1d(Lambda_sel_tmp, Lam_comn, invert=True)][:sprsty]       
                elif S_comn==0: 
                    Lambda_sel_tmp = Lambda_sel_tmp[:sprsty]
                print("Lambda_sel_tmp",Lambda_sel_tmp)
                Lambda_sel = np.union1d(Lambda_sel_tmp,Lam_pr_bst)
                print("Lambda_sel",Lambda_sel)

            #import pdb; pdb.set_trace()
            Psi_active_bst = pcu.make_Psi_drn(y_data[optim_indices,:d],mi_mat,Lambda_sel.tolist(),chc_Psi)
            Psi_active_bst_T = np.transpose(Psi_active_bst)
            c_hat_bst = (la.inv(Psi_active_bst_T @ Psi_active_bst) @ Psi_active_bst_T @ u_data[optim_indices]).flatten()
            c_omp_sel[Lambda_sel] = c_hat_bst
            Lambda_bst = (np.argsort(np.abs(c_omp_sel))[::-1])[:sprsty]
            c_omp_bst[Lambda_bst] = c_omp_sel[Lambda_bst]
#FIXME do least squares using Lambda_bst instead of selecting from the already chosen one:  
            test_err_bst, valid_err_bst = tnn.val_test_err(data_tst,mi_mat,c_omp_bst)
#%%====================calculate the omp coefficients on the residual signal================
            Lambda_bst_mp = [i_mp for i_mp,vl_lmbst in enumerate(Lambda_sel) if vl_lmbst in Lambda_bst]
            print('Lambda_bst',Lambda_bst)
            print('Lambda_sel',Lambda_sel)
            print('Lambda_bst_mp',Lambda_bst_mp)
#            import pdb; pdb.set_trace()            
            rsdl =  u_data[optim_indices] - Psi_active_bst[:,Lambda_bst_mp] @ c_omp_bst[Lambda_sel[Lambda_bst_mp]]
#            Psi_omp = pcu.make_Psi(y_data[optim_indices,:d],mi_mat_omp,chc_Psi) 
            #import pdb;pdb.set_trace()
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
            trn_ind_nw = np.concatenate((cr_mxind,np.array(best_config.get("tind_nz")),np.array(best_config.get("tind_z"))),axis=None)
            df_trn_ind_nw = pd.DataFrame({'trn_ind_nw':trn_ind_nw})        
            df_trn_ind_nw.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/trn_indices_alph_omp_N={N}_{i}_c{trc}.csv',index=False)
            val_ind_nw = np.setdiff1d(np.linspace(0,P_alg-1,P_alg),trn_ind_nw)
            df_val_ind_nw = pd.DataFrame({'val_ind_nw':val_ind_nw})        
            df_val_ind_nw.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/val_indices_alph_omp_N={N}_{i}_c{trc}.csv',index=False)
            #%% Write stuff:
            df_b_params = pd.DataFrame({key_cfg:val_cfg for key_cfg,val_cfg in best_config.items() if key_cfg not in ['tind_nz','tind_z']},index=[0])
            df_b_params.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/Best_hyper_params_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)           
            df_Lam_sel = pd.DataFrame({'Lam_sel':Lambda_sel})
            df_Lam_sel.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/Lam_sel_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)           
            df_Lam_bst = pd.DataFrame({'Lam_bst':Lambda_bst})
            df_Lam_bst.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/Lam_bst_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)           
            df_c_omp_rs = pd.DataFrame({'comp_rs':c_om_rs})
            df_c_omp_rs.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/comp_rs_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)               
            # cost_tot_1 = cost_tot[1:,:]
            df_bepoch = pd.DataFrame({'ep_best':best_epoch[-1]},index=[0])
            df_bepoch.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/best_epoch_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)
            df_cost_tot = pd.DataFrame({'cost_t':np.array(cost)})
            df_cost_tot.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/cost_tot_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)
            df_cost_val = pd.DataFrame({'cost_val':np.array(cost_val)})
            df_cost_val.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/cost_val_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)
            # df_cost_val = pd.DataFrame(np.array(cost_val))
            # df_cost_val.to_csv(f'{out_dir_ini}/plots/cost_val_1dellps_n={N}_genmod_S={sprsty}.csv',index=False)
            df_c_hat = pd.DataFrame({'c_hat':c_hat})
            df_c_hat.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/c_hat_tot_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)
            df_Gs = pd.DataFrame({'Gmod_bst':Gmod_bst})
            df_Gs.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/Gmod_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)
            df_c_omp_bst = pd.DataFrame({'comp_fnl':c_omp_bst})
            df_c_omp_bst.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/comp_fl_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)
            df_c_omp_sel = pd.DataFrame({'comp_sel':c_omp_sel})
            df_c_omp_sel.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/comp_sel_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)
            df_cini = pd.DataFrame({'cini':c_ini})
            df_cini.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/cini_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)
            # df_eps_ctmp = pd.DataFrame({'eps_ctmp':eps_ctmp}) 
            # df_eps_ctmp.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/eps_ctmp_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)
            df_thet_up = pd.DataFrame({'thet_bst':thet_bst})
            df_thet_up.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/thetup_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)  
          #df_thet_up = pd.DataFrame({'thet_f':thet_f,'thet_bst':thet_bst})
          #df_thet_up.to_csv(f'{out_dir_ini}/plots/j={j}/it={i}/thetup_1dellps_n={N}_genmod_S={sprsty}_{i}_j{j}_c{trc}.csv',index=False)  

        # if i==0:
        #     ecmn_ind = np.argmin(eps_ctmp)
        #     eps_c[i+1] = eps_ctmp[ecmn_ind]
        # else:
        #     ecmn_ind[i] = np.argmin(eps_ctmp)
        #     eps_c[i+1] = eps_ctmp[int(ecmn_ind[i])]
        if chc_eps == 'c':
            ecmn_ind[i] = np.argmin(eps_ctmp)
            eps_c[i+1] = eps_ctmp[int(ecmn_ind[i])]
        elif chc_eps == 'u': 
            ecmn_ind[i] = np.argmin(epsu_tmp)
            eps_u[i+1] = epsu_tmp[int(ecmn_ind[i])]            
        # # Testing error: is the error on the training data:
        # print(f'Testing Error: {test_err_bst}')
        # #=================================================================
        # # #Log plot of cini and GMod comparision (basically nonzeros):
        # plt.figure(7+5*i)
        # plt.semilogy(np.linspace(1,P,P),Gmod_bst,'ok',label='G_1',markersize=10)
        # # plt.semilogy(np.linspace(1,P_omp,P_omp),np.abs(c_ini),'*r',label='|C_omp|')
        # plt.semilogy(np.linspace(1,P,P),np.abs(c_ref),'*r',label='|C_omp|',markersize=5)
        # # plt.ylim([1e-7,0.4])
        # plt.xlabel('Index of PCE coefficients')
        # plt.ylabel('Magnitude of PCE coefficients')
        i += 1
        # #=================================================================
    #%% Plot relative validation error:
    # Wrtite relative coefficient error:
    if chc_eps == 'u':
        df_epsu = pd.DataFrame({'eps_u':eps_u})
        df_epsu.to_csv(f'{out_dir_ini}/plots/j={j}/epsu_1dellps_n={N}_genmod_S={sprsty}_j{j}.csv',index=False)  
        # Wrtite relative coefficient error:
    elif chc_eps == 'c': 
        df_epsc = pd.DataFrame({'eps_c':eps_c})
        df_epsc.to_csv(f'{out_dir_ini}/plots/j={j}/epsc_1dellps_n={N}_genmod_S={sprsty}_j{j}.csv',index=False)        
    df_mnd= pd.DataFrame({'ecmn_ind':ecmn_ind})
    df_mnd.to_csv(f'{out_dir_ini}/plots/j={j}/ecmn_ind_1dellps_n={N}_genmod_S={sprsty}_j{j}.csv',index=False)
    # Wrtite weighted coefficient error:
    df_epsc_abs = pd.DataFrame({'epsc_abs':eps_abs})
    df_epsc_abs.to_csv(f'{out_dir_ini}/plots/j={j}/epsc_abs_1dellps_n={N}_genmod_S={sprsty}_j{j}.csv',index=False)        
    if chc_eps=='c':
        df_cref = pd.DataFrame({'c_ref':c_ref})
        df_cref.to_csv(f'{out_dir_ini}/plots/j={j}/c_ref_1dellps_n={N}_genmod_S={sprsty}_j{j}.csv',index=False)
    #df_epscomp = pd.DataFrame({'epsu_omph':valid_omp_ph,'epsu_omph_t':test_omp_ph},index=[0])
    #df_epscomp.to_csv(f'{out_dir_ini}/plots/epsuomph_tst_1dellps_n={N}_genmod_S={sprsty}_j{j}.csv',index=False)
#plt.show()
end_time = time.time()
print('end - start times:',end_time-start_time)

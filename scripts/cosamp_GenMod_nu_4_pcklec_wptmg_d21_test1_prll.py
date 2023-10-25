# -*-0 coding: utf-8 -*-
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
import multiprocessing
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
import genmod_mod_test.test_coeffs_val_er_utils as tcu
import genmod_mod_test.mo_main_fn as mmf
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
parser.add_argument('--Nt',dest='N_t',default=5,type=int,help='Number of total iterations')
parser.add_argument('--Nrep',dest='N_rep',default=1,type=int,help='Number of sample replications')
parser.add_argument('--Nlhid',dest='N_hid',default=1,type=int,help='Number of sample replications')
parser.add_argument('--epochs',dest='ep',default=15000,type=int,help='Number of epochs')
parser.add_argument('--iter_fix',dest='it_fix',default=9999,type=int,help='After this number the metric is evaluated')
parser.add_argument('--nproc',dest='num_work',default=1,type=int,help='After this number the metric is evaluated')
parser.add_argument('--sd_j',dest='sd_indic_ind',default=1,type=int,help='After this number the metric is evaluated')
parser.add_argument('--p_h',dest='ph',default=3,type=int,help='p_h')
parser.add_argument('--case',dest='case_ppr',default='ttn_78',type=str,help='ttn_78, ttn_21, 1dell_14')
parser.add_argument('--hbd',dest='h_bnd',nargs='+',default=[7,8],type=int,help='p_h')
parser.add_argument('--p_0',dest='p0',default=2,type=int,help='p_0')
parser.add_argument('--top_i1',dest='topi1',default=2,type=int,help='p_0')
parser.add_argument('--top_i0',dest='topi0',default=3,type=int,help='p_0')
parser.add_argument('--vlcf_add',dest='vlcfadd',default=0,type=int,help='p_0')
parser.add_argument('--d',dest='dim',default=21,type=int,help='d')
parser.add_argument('--dbgit2',dest='dbg_it2',default=0,type=int,help='change after the second iteration')
parser.add_argument('--S_chs',dest='chs_sprs',default=0,type=int,help='S_chs')
parser.add_argument('--res_tol',dest='resomp_tol',default=1e-10,type=float,help='p_h')
parser.add_argument('--ompsol',dest='chc_omp',default='stdomp',type=str,help='p_h')
parser.add_argument('--poly',dest='chc_poly',default='Hermite',type=str,help='p_h')
parser.add_argument('--qoi',dest='QoI',default='heat_flux',type=str,help='quantity of Interest')
parser.add_argument('--use_gmd',dest='use_gmd',default=0,type=int,help='d')
parser.add_argument('--omponly',dest='omp_only',default=0,type=int,help='switch to use only OMP calculations')
parser.add_argument('--so_res',dest='add_tpso_res',default=0,type=int,help='switch to use only OMP calculations')
#parser.add_argument('--cini_fl',dest='cht_ini_fl',default='/home/jothi/CoSaMP_genNN/output/titan_ppr/results/csaug13/d=21/p=3/ref_dbg/ttne_913_j2smx2S_Nt5/plots/j=1/it=3/c_hat_tot_1dellps_n=100_genmod_S=7_3_j1_c0.csv',type=str,help='file name with the path for initial omp coefficients for reproducing/debugging')
parser.add_argument('--cini_fl',dest='cht_ini_fl',default='/home/jothi/CoSaMP_genNN/output/titan_ppr/results/d78_ppr/ref_dbg/cls_1dellps_n=5000_genmod_S=7_p=3_j0.csv',type=str,help='file name with the path for initial omp coefficients for reproducing/debugging')
parser.add_argument('--ntrial',dest='num_trl',default=10,type=int,help='num_trials')
parser.add_argument('--S_fac',dest='mul_fac',default=2,type=int,help='1-flag for switching to debugging')
parser.add_argument('--dbg',dest='debug_alg',default=0,type=int,help='1-flag for switching to debugging')
parser.add_argument('--pltj',dest='plt_spcdt',default=-1,type=int,help='>0-flag for switching to any other sample')
parser.add_argument('--dbg_act',dest='debug_act',default=0,type=int,help='1-flag for Relu alone,2-flag for None alone ')
parser.add_argument('--tind',dest='dbg_rdtvind',default=0,type=int,help='1-flag for reading train/valid indices from file')
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
S_chs = args.mul_fac*S_omp
freq = 1 
tot_itr = args.N_t
Nc_rp = 1 # NOTE: Set this as always 1 for this particular case.
Nrp_vl = 1
ecmn_ind = 0
W_fac = np.ones(tot_itr)
if args.chs_sprs==0:
    sprsty = S_omp
else:
    sprsty = int(input("enter the extra active basis functions to be included in addition to the Sh"))

chc_eps = 'u'
chc_Psi = args.chc_poly #'Hermite'
chc_omp_slv= args.chc_omp #'stdomp'#'ompcv' #'stdomp' #FIXME  
tune_sg = args.tune_sig
pltdta = args.plot_dat #switch to 1 if data should be plotted.
top_i1 = args.topi1 #int(4*cini_nz_ln/5-ntpk_cr), ntpk_cr = top_i1. 4*cini_nz_ln/5 should be > ntpk_cr
top_i0 = args.topi0 
#Seed values:
seed_ind = args.sd_indic_ind
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
if args.case_ppr=="ttn_78":
    # Read for the Maximum CN concentration for the previous data:
    out_dir_sct = '../data/Titn_rcnt_dta/dx1em3/LN_d78_hghunkF'    
    #y_data1 = pd.read_csv(f'{out_dir_sct}/files/Aval_zis_frm_invmp_78d_N20k.csv').to_numpy() #incorrect file
    y_data1 = pd.read_csv(f'{out_dir_sct}/files/Aval_zis_frm_invmp_78_ln_hghunkF.csv').to_numpy()
    y_data = y_data1[:,0:d] 
    u_data2 = pd.read_csv(f'{out_dir_sct}/xCN/xCN_smp.csv').to_numpy()
    u_data1 = np.transpose(u_data2[:,2:]) #First two colums are just junks.
    u_data = np.amax(u_data1,axis=1)
    print('u_data1[:5,:5]',u_data1[:5,:5])
    print('shape of u_data1:',u_data1.shape)
#===========================================================================
#===========================================================================
elif args.case_ppr=="1dell_14":
    chc_Psi = 'Legendre' #'Hermite'
    data = sio.loadmat('../data/dataset1/1Dellptic_data.mat')
    y_data = data['y']
    u_data = data['u'].flatten()
#===========================================================================
elif args.case_ppr=="ttn_16":
    # Read for the Maximum CN concentration for the previous data:
    #Maximum CN concentration: 
    out_dir_sct = '../data/Titn_rcnt_dta/d16' 
    #y_data1 = pd.read_csv(f'{out_dir_sct}/files/Aval_zis_frm_invmp_78d_N20k.csv').to_numpy() #incorrect file
    y_data1 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N10000_d16.csv').to_numpy()
    y_data = y_data1[:,0:d] 
    u_data1 = pd.read_csv(f'{out_dir_sct}/xCN_smp.csv').to_numpy()
    u_data = np.amax(u_data1,axis=0)
    print("Choice of polynomial required: Hermite")    
    print(f"Choice of polynomial used: {chc_Psi}")    
    print('u_data1[:5,:5]',u_data1[:5,:5])
    print('shape of u_data1:',u_data1.shape)
    import pdb; pdb.set_trace()
elif args.case_ppr=="ttn_20":
    # Read for the Maximum CN concentration for the previous data:
    #out_dir_sct = '../data/Titn_rcnt_dta/opLNtp20_sm200'    
    ##y_data1 = pd.read_csv(f'{out_dir_sct}/files/Aval_zis_frm_invmp_78d_N20k.csv').to_numpy() #incorrect file
    #y_data1 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N1000_d20.csv').to_numpy()
    #y_data = y_data1[:,0:d] 
    #x_data = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000.csv').to_numpy()
    #u_data1 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_PLATO_to_MURP_N1000.csv').to_numpy()
    #u_data = np.mean(u_data1,axis=0)
    #Maximum CN concentration: 
    out_dir_sct = '../data/Titn_rcnt_dta/d20' 
    #y_data1 = pd.read_csv(f'{out_dir_sct}/files/Aval_zis_frm_invmp_78d_N20k.csv').to_numpy() #incorrect file
    y_data1 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N10000_d20.csv').to_numpy()
    y_data = y_data1[:,0:d] 
    u_data1 = pd.read_csv(f'{out_dir_sct}/xCN_smp.csv').to_numpy()
    u_data = np.amax(u_data1,axis=0)
    print('u_data1[:5,:5]',u_data1[:5,:5])
    print('shape of u_data1:',u_data1.shape)
    import pdb; pdb.set_trace()
#===========================================================================
# read data for MURP outputs: 
elif args.case_ppr=="ttn_21":
    out_dir_sct = '../data/Titn_rcnt_dta/d21'    
    ##u_data1 has a shape of "Number of grid points by N_samp" for Wall-directed radiative heat flux.
    y_data11 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N1000_d21_sd100_plsU4_CH4.csv').to_numpy()
    y_data12 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N1000_d21_sd200_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    y_data13 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N8000_d21_sd300_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    y_data13 = y_data13[:6000,:d] #FIXME
    y_data14 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N4000_d21_sd400_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    y_data15 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N2000_d21_sd500_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    y_data1 = np.vstack((y_data11,y_data12,y_data13,y_data14,y_data15))
    y_data1[:,d-1] = norm.ppf(0.5*(y_data1[:,d-1]+1))
    y_data = y_data1[:,0:d] 
    u_data11 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_PLATO_to_MURP_N1000.csv').to_numpy()
    u_data12 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N1000_ext1k.csv').to_numpy()
    u_data13 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N6000.csv').to_numpy()
    u_data14 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N6000_ext6k_4kp2k.csv').to_numpy()
    u_data1 = np.hstack((u_data11,u_data12,u_data13,u_data14))
    u_data = np.mean(u_data1,axis=0)
    x_data11 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000.csv').to_numpy()
    x_data12 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000_ext1k.csv').to_numpy()
    x_data13 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_Plato_to_MURP_N6000.csv').to_numpy()
    x_data14 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_Plato_to_MURP_N6000_ext6k_4kp2k.csv').to_numpy()
    #x_data = np.hstack((x_data11,x_data12,x_data13)) #FIXME
    x_data = np.hstack((x_data11,x_data12,x_data13,x_data14))
    ##===========================================================================
    ##===========================================================================
    
    A_data11 = pd.read_csv(f'{out_dir_sct}/files/Aval_fl_4plto_LN_N1000_d21_sd100_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data12 = pd.read_csv(f'{out_dir_sct}/files/Aval_fl_4plto_LN_N1000_d21_sd200_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data13 = pd.read_csv(f'{out_dir_sct}/files/Aval_fl_4plto_LN_N8000_d21_sd300_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data13 = A_data13[:6000,:d] #FIXME
    A_data14 = pd.read_csv(f'{out_dir_sct}/files/Aval_fl_4plto_LN_N4000_d21_sd400_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data15 = pd.read_csv(f'{out_dir_sct}/files/Aval_fl_4plto_LN_N2000_d21_sd500_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data = np.vstack((A_data11,A_data12,A_data13,A_data14,A_data15))
    ##import pdb; pdb.set_trace()
    ##u_data1 = np.hstack((u_data11,u_data12,u_data13)) #FIXME
    #u_data = np.mean(u_data1,axis=0)
    #y_data = y_data1[:,0:d] 
    #print('u_data1[:5,:5]',u_data1[:5,:5])
    #print('shape of u_data1:',u_data1.shape)
    #To run on maximum CN concentration data:    
    #y_data13 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N8000_d21_sd300_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    #y_data14 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N4000_d21_sd400_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    #y_data15 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N2000_d21_sd500_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
   # y_data1 = np.vstack((y_data13,y_data14,y_data15))
   #     #y_data1 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N8000_d21_sd300_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
   # #y_data1 = np.vstack((y_data11,y_data12,y_data13)) #FIXME
   # #===========================================================================
   # #===========================================================================
   # y_data1[:,d-1] = norm.ppf(0.5*(y_data1[:,d-1]+1))
    if args.QoI!='heat_flux':
        print("Maximum CN concentration is used as QoI")    
        u_data13 = pd.read_csv(f'{out_dir_sct}/xCN_smp_N8k.csv').to_numpy()
        u_data14 = pd.read_csv(f'{out_dir_sct}/xCN_smp_N4k.csv').to_numpy()
        u_data15 = pd.read_csv(f'{out_dir_sct}/xCN_smp_N2k.csv').to_numpy()
        u_data1 = np.hstack((u_data13,u_data14,u_data15))
        u_data = np.amax(u_data1,axis=0)
        #u_data1 = pd.read_csv(f'{out_dir_sct}/xCN_smp_N8k.csv').to_numpy()
    #u_data1 = np.hstack((u_data11,u_data12,u_data13)) #FIXME
    #x_data11 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000.csv').to_numpy()
    #x_data12 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000_ext1k.csv').to_numpy()
    #x_data13 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_Plato_to_MURP_N6000.csv').to_numpy()
    #x_data14 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_Plato_to_MURP_N6000_ext6k_4kp2k.csv').to_numpy()
    ##x_data = np.hstack((x_data11,x_data12,x_data13)) #FIXME
    #x_data = np.hstack((x_data11,x_data12,x_data13,x_data14))
    print('u_data1[:5,:5]',u_data1[:5,:5])
    print('shape of u_data1:',u_data1.shape)
    import pdb; pdb.set_trace()
elif args.case_ppr=="ttn_21_dbg":
    out_dir_sct = '../data/Titn_rcnt_dta/d21'    
    ##u_data1 has a shape of "Number of grid points by N_samp" for Wall-directed radiative heat flux.
    y_data11 = pd.read_csv(f'{out_dir_sct}/d21_dbg/files/ynrm_fl_stdGsn_N12000_d21_sd100_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    y_data12 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N1000_d21_sd200_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    y_data13 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N8000_d21_sd300_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    y_data13 = y_data13[:6000,:d] #FIXME
    y_data14 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N4000_d21_sd400_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    y_data15 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N2000_d21_sd500_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    y_data1 = np.vstack((y_data11,y_data12,y_data13,y_data14,y_data15))
    y_data1[:,d-1] = norm.ppf(0.5*(y_data1[:,d-1]+1))
    y_data = y_data1[:,0:d] 
    u_data11 = pd.read_csv(f'{out_dir_sct}/d21_dbg/MURP_data/Qrad_dat_Plato_to_MURP_N12000.csv').to_numpy()
    u_data12 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N1000_ext1k.csv').to_numpy()
    u_data13 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N6000.csv').to_numpy()
    u_data14 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N6000_ext6k_4kp2k.csv').to_numpy()
    u_data1 = np.hstack((u_data11,u_data12,u_data13,u_data14))
    u_data = np.mean(u_data1,axis=0)
    x_data11 = pd.read_csv(f'{out_dir_sct}/d21_dbg/MURP_data/x_dat_Plato_to_MURP_N12000.csv').to_numpy()
    x_data12 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000_ext1k.csv').to_numpy()
    x_data13 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_Plato_to_MURP_N6000.csv').to_numpy()
    x_data14 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_Plato_to_MURP_N6000_ext6k_4kp2k.csv').to_numpy()
    #x_data = np.hstack((x_data11,x_data12,x_data13)) #FIXME
    x_data = np.hstack((x_data11,x_data12,x_data13,x_data14))
    ##===========================================================================
    ##===========================================================================
    
    A_data11 = pd.read_csv(f'{out_dir_sct}/d21_dbg/files/Aval_fl_4plto_LN_N12000_d21_sd100_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data12 = pd.read_csv(f'{out_dir_sct}/files/Aval_fl_4plto_LN_N1000_d21_sd200_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data13 = pd.read_csv(f'{out_dir_sct}/files/Aval_fl_4plto_LN_N8000_d21_sd300_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data13 = A_data13[:6000,:d] #FIXME
    A_data14 = pd.read_csv(f'{out_dir_sct}/files/Aval_fl_4plto_LN_N4000_d21_sd400_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data15 = pd.read_csv(f'{out_dir_sct}/files/Aval_fl_4plto_LN_N2000_d21_sd500_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data = np.vstack((A_data11,A_data12,A_data13,A_data14,A_data15))
elif args.case_ppr=="ttn_21_nw":
    # Read for the Maximum CN concentration for the previous data:
    #out_dir_sct = '../data/Titn_rcnt_dta/opLNtp20_sm200'
    ##y_data1 = pd.read_csv(f'{out_dir_sct}/files/Aval_zis_frm_invmp_78d_N20k.csv').to_numpy() #incorrect file
    #y_data1 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N1000_d20.csv').to_numpy()
    #y_data = y_data1[:,0:d]
    #x_data = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000.csv').to_numpy()
    #u_data1 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_PLATO_to_MURP_N1000.csv').to_numpy()
    #u_data = np.mean(u_data1,axis=0)
    #Maximum CN concentration:
    out_dir_sct = '../data/Titn_rcnt_dta/d21_dbg'
    #y_data1 = pd.read_csv(f'{out_dir_sct}/files/Aval_zis_frm_invmp_78d_N20k.csv').to_numpy() #incorrect file
    y_data1 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N12000_d21_sd100_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    y_data1[:,d-1] = norm.ppf(0.5*(y_data1[:,d-1]+1))
    y_data = y_data1[:,0:d]

    #u_data1 = pd.read_csv(f'{out_dir_sct}/xCN_smp.csv').to_numpy()
    #u_data = np.amax(u_data1,axis=0)
    print('u_data1[:5,:5]',u_data1[:5,:5])
    print('shape of u_data1:',u_data1.shape)
elif args.case_ppr=="ttn_79":
    out_dir_sct = '../data/Titn_rcnt_dta/d79'    
    y_data1 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N10000_d79_sd100_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    y_data1[:,d-1] = norm.ppf(0.5*(y_data1[:,d-1]+1))
    y_data = y_data1[:,0:d] 
    x_data = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_Plato_to_MURP_N10000.csv').to_numpy()
    u_data1 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N10000.csv').to_numpy()
    u_data = np.mean(u_data1,axis=0)
   #Maximum CN concentration:
   # x_data = pd.read_csv(f'{out_dir_sct}/x_smp.csv').to_numpy()
   # u_data1 = pd.read_csv(f'{out_dir_sct}/xCN_smp.csv').to_numpy()
   # u_data = np.amax(u_data1,axis=0)
    print('u_data1[:5,:5]',u_data1[:5,:5])
    print('shape of u_data1:',u_data1.shape)
elif args.case_ppr=="ttn_78_sjo":  
    chc_Psi = 'Legendre' #'Hermite'
    out_dir_sct = '../data/Titan_sjo_unium/d78'    
    data = sio.loadmat(f'{out_dir_sct}/Jul16_data.mat')
    y_data = data['Y']
    u_data1 = data['U']
    u_data = np.amax(u_data1,axis=1) 
    print("Choice of polynomial required: Legendre")    
    print(f"Choice of polynomial used: {chc_Psi}")    
    print('u_data1[:5,:5]',u_data1[:5,:5])
    print('shape of u_data1:',u_data1.shape)
elif args.case_ppr=="ttn_16_sjo":  
    chc_Psi = 'Legendre' #'Hermite'
    out_dir_sct = '../data/Titan_sjo_unium/d16'    
    data = sio.loadmat(f'{out_dir_sct}/Jul28_data_d16.mat')
    y_data = data['Y']
    u_data1 = data['U']
    u_data = np.amax(u_data1,axis=1) 
    print("Choice of polynomial required: Legendre")    
    print(f"Choice of polynomial used: {chc_Psi}")    
    #import pdb; pdb.set_trace()
    print('u_data1[:5,:5]',u_data1[:5,:5])
    print('shape of u_data1:',u_data1.shape)
    print('shape of u_data:',u_data.shape)
elif args.case_ppr=="mnfct_dt":  
    out_dir_sct = '../data/mnfctrd_data/S=10'    
    y_data = pd.read_csv(f'{out_dir_sct}/y_data_rnd_gaussian_d20_seed1.csv').to_numpy()
    u_data = pd.read_csv(f'{out_dir_sct}/u_mnfct_1dellps_n=5000_genmodNN_exp_p=3_S=10.csv').to_numpy().flatten()
    print("Choice of polynomial required: Hermite")    
    print(f"Choice of polynomial used: {chc_Psi}")    
    #print('u_data1[:5,:5]',u_data1[:5,:5])
elif args.case_ppr=="mnfct_dts4":  
    out_dir_sct = '../data/mnfctrd_data/S=4'    
    y_data = pd.read_csv(f'{out_dir_sct}/y_data_rnd_gaussian_d20_seed1.csv').to_numpy()
    u_data = pd.read_csv(f'{out_dir_sct}/u_mnfct_1dellps_n=5000_genmodNN_exp_p=3_S=4.csv').to_numpy().flatten()
    print("Choice of polynomial required: Hermite")    
    print(f"Choice of polynomial used: {chc_Psi}")    
elif args.case_ppr=="sqr_cvty_20":  
    chc_Psi = 'Legendre' #'Hermite'
    out_dir_sct = '../data/dataset2'    
    u_data1 = sio.loadmat(f'{out_dir_sct}/u_samples.mat') 
    y_data1 = sio.loadmat(f'{out_dir_sct}/y_samples.mat')
    y_data = y_data1['y_samples']
    u_data = u_data1['u_samples'].flatten()
    print("Choice of polynomial required: Legendre")    
    print(f"Choice of polynomial used: {chc_Psi}")    
    #print('u_data1[:5,:5]',u_data1[:5,:5])

#import pdb; pdb.set_trace()
#===========================================================================
#===========================================================================
#import pdb;pdb.set_trace()
print('shape of u_data:',u_data.shape)
print('shape of y_data:',y_data.shape)
#TODO
print('=====================================================================')
print('=========Check the independence of the distributions=================')
print('=====================================================================')
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
    u_plt_rnd = random.sample(range(2000),200) #FIXME.
    #import pdb; pdb.set_trace()
    for i in u_plt_rnd:
            #plt.plot(x_data[:,i],u_data1[:,i])
            plt.plot(u_data1[:,i])
            #plt.plot(u_data[:,i])
            #plt.plot(u_data1[i,:])
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
mi_mat = pcu.make_mi_mat(d, p)
mi_mat_p0 = pcu.make_mi_mat(d, p_0)
df_mimat = pd.DataFrame(mi_mat)
df_mimat.to_csv(f'{out_dir_ini}/mi_mat_pd={p,d}.csv',index=False)
P = np.size(mi_mat,0)  
data_all = {'y_data':y_data,'u_data':u_data,'mi_mat':mi_mat} 
#print('n',n)
#%% initial parameters:
# sprsty = 43
# sprsty = 5  #Think about giving sparsity=1, some matrix manipulations might get affected.
learning_rate = 0.001
epochs = args.ep
#avtnlst =['None' for a_m in range(Nlhid)]#[nn.Sigmoid()] # for the final layer by default exp decay is enforced, so the size is number of layers-1.
if args.debug_act==1:
    avtnlst =['None'  if tune_sg==0 else tune.choice([nn.ReLU()]) for a_m in range(Nlhid)] #[nn.Sigmoid()] # for the final layer by default exp decay is enforced, so the size is number of layers-1.
elif args.debug_act==2:
    avtnlst =['None'  if tune_sg==0 else tune.choice(['None']) for a_m in range(Nlhid)]    
else:
    avtnlst =['None'  if tune_sg==0 else tune.choice(['None',nn.Sigmoid(),nn.ReLU()]) for a_m in range(Nlhid)] #[nn.Sigmoid()] # for the final layer by default exp decay is enforced, so the size is number of layers-1.
#import pdb; pdb.set_trace()
#avtnlst = ['None',nn.Sigmoid(),nn.Relu()]
#%% Write index file:
N = args.N_smp  # note that you need more samples than sparsity for least squares.
Nv = args.N_v
Nrep = args.N_rep
j_rng = range(Nrep) if args.j_flg==0 else args.j_flg #range(Nrep) ---change this to run for a particular replication. Useful for debugging.
print('N:',N,'Nv:',Nv)
#% Save parameters:
opt_params = {'ph':p,'p0':p_0,'d':d,'P':P,'epochs':epochs,'lr':learning_rate,'Sh':S_omp,'S0':S_omp0, 'sprsty':sprsty,
        'N_t':tot_itr,'fr':freq,'W_fac':f'{W_fac}','z_n':z_n,'Tp_i1':top_i1,'Tp_i0':top_i0,'N':N,'Nv':Nv,'Nrep':Nrep,'Nc_rp':Nc_rp,'S_chs':S_chs,'chc_poly':chc_Psi,'sd_ind':seed_ind,'sd_thtini':seed_thtini,'sd_ceff':seed_ceff,'Nrp_vl':Nrp_vl,"sd_thtini_2nd":sd_thtini_2nd,'iter_fix':args.it_fix,'ntrial':args.num_trl,'Nlhid':Nlhid,'chc_omp_slv':chc_omp_slv,'chc_eps':chc_eps}
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
nn_prms_dict = {'avtnlst':avtnlst,'hid_layers':hid_layers,'tune_sg':tune_sg}
#
if args.plt_spcdt>=0:
    j_plt =args.plt_spcdt    
    optim_ind = indices0.iloc[j_plt].to_numpy()
    plt.figure(4)
    plt.plot(np.arange(1,N+1),u_data[optim_ind])
    plt.xlabel('sample index')
    plt.ylabel('Mean radiative heat flux[W/cm^2]')
    plt.savefig(f"{out_dir_ini}/u_data_mean_j{j_plt}.png",dpi=300)
#
   # plt.figure(1)
   # #u_plt_rnd = random.sample(range(0,np.size(u_data)), 20)
   # #u_plt_rnd = random.sample(range(2000,np.size(u_data)), N_y-2000) #FIXME.
   # #import pdb; pdb.set_trace()
   # for i in optim_ind:
   #         plt.plot(x_data[:,i],u_data1[:,i])
   #         #plt.plot(u_data1[:,i])
   #         #plt.plot(u_data[:,i])
   #         #plt.plot(u_data1[i,:])
   #         #plt.plot(u_data1[:,i],label=f'i={i}')
   #         plt.xlabel('spatial location x')
   #         plt.ylabel('Radiative heat flux[W/cm^2]')
   # plt.savefig(f"{out_dir_ini}/u_data_j{j_plt}.png",dpi=300)
   # #%% Visualize ydata:
   # #==============================================================================
   # #==============================================================================
   # # Plot scatter plot of input parameters (Jacqui)
   # plt.figure(2)
   # y_df = pd.DataFrame(y_data)
   # y_df.shape
   # sns.pairplot(y_df.iloc[optim_ind, :d], kind="scatter")
   # plt.savefig(f"{out_dir_ini}/pair_plot_y_j{j_plt}.png") #,dpi=300)
   # plt.figure(3)
   # A_df = pd.DataFrame(A_data)
   # A_df.shape
   # sns.pairplot(A_df.iloc[optim_ind, :d], kind="scatter")
   # plt.savefig(f"{out_dir_ini}/pair_plot_A_j{j_plt}.png") #,dpi=300)
   # import pdb; pdb.set_trace()
        
#plot histograms
    #num_bins = 20  # You can adjust this as needed

    ## Create a single figure with subplots for each random variable
    #fig, axes = plt.subplots(5, int(num_variables/5)+1, figsize=(15, 5))  # Adjust figsize as needed

    #for i in range(num_variables):
    #    axes[i].hist(data_matrix[:, i], bins=num_bins, edgecolor='k')
    #    axes[i].set_title(f'Random Variable {i + 1}')
    #    axes[i].set_xlabel('Value')
    #    axes[i].set_ylabel('Frequency')

    #plt.tight_layout()  # Helps prevent overlap of titles and labels
    #plt.show()

#==============================================================================
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
#=================================================================================
#for testing purposes: test the sparsity values for different sample replications 
#=================================================================================
#S_rep_test = []; S_rep_test_ph = []
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
#    c_ini, S_omp0, train_err_p0, valid_err_p0,P_omp,mi_mat_omp, Psi_omp = omu.omp_utils_order_ph(out_dir_ini,d,p_0,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_omp0,j)
#    c_omph, S_omph, test_omp_ph, valid_omp_ph,P_omph,mi_mat_omph, Psi_omph= omu.omp_utils_order_ph(out_dir_ini,d,p,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_omp,j)
#    S_rep_test.append(S_omp0)
#    S_rep_test_ph.append(S_omph)
#S_rep_avg = sum(S_rep_test)/len(S_rep_test)    
#df_S_reptest = pd.DataFrame({'S_lst':np.array(S_rep_test)})
#df_S_reptest.to_csv(f'{out_dir_ini}/plots/S_rep_list.csv',index=False)
#df_S_reptest_ph = pd.DataFrame({'S_lst':np.array(S_rep_test_ph)})
#df_S_reptest_ph.to_csv(f'{out_dir_ini}/plots/S_rep_list_ph.csv',index=False)
#=================================================================================
#======================================================================================
#============Apply least squares for the particular active set=========================
#======================================================================================
# Do not use the parallel calculation of Psi for high dimensional problems: it uses almost all memory 
# for 79 dimension problem.
#for j in j_rng:
#    print(f'=============#replication={j}============')
#    ecmn_ind = np.zeros(tot_itr)
#    optim_indices = indices0.iloc[j].to_numpy()
#    valid_indices = np.setdiff1d(range(np.size(u_data)), optim_indices)
#    trains = [name for name in indices0.columns if name.startswith("optim")]
#    test_indices = indices0.loc[j][trains].to_numpy()
#    data_tst = {'y_data':y_data,'u_data':u_data,'val_ind':valid_indices,'test_ind':test_indices,'opt_ind':optim_indices,'Nv':Nv,
#            'chc_poly':chc_Psi,'chc_omp':chc_omp_slv} 
#    c_omph, S_omph, test_omp_ph, valid_omp_ph,P_omph,mi_mat_omph, Psi_omph= omu.omp_utils_order_ph_prllNcalc(out_dir_ini,d,p,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_omp,j)
#import pdb; pdb.set_trace()
#=================================================================================
#======================================================================================
## Testing some leasrt squares solution:
#    val_ls_tot = []        
#    trn_ls_tot = []        
#    S_strt = 7     
#    S_true_78 = [0,60,2,1,74,9,51,3145,48,157,12,63,5,85285,21,229,49,726,80,138,4,79,152,
#                        2664,7,70,2984,2763,9229,714,2754,1449,305,215,11,13,1479,36,2690,218]        
#    print('S_true_78',S_true_78)    
#    for ls_ind in range(len(S_true_78)-S_strt+1): 
#        c_ls = tcu.apply_lst_sqr_actset(S_true_78[:S_strt+ls_ind],P,mi_mat,chc_Psi,u_data,y_data,optim_indices)    
#        trn_err_ls, valid_err_ls = tnn.val_test_err(data_tst,mi_mat,c_ls)
#        val_ls_tot.append(valid_err_ls)
#        trn_ls_tot.append(trn_err_ls)
#    df_err_ls = pd.DataFrame({'val_ls':np.array(val_ls_tot),'trn_ls':np.array(trn_ls_tot)})
#    df_err_ls.to_csv(f'{out_dir_ini}/plots/trnval_ls_list_j{j}.csv',index=False)
#    plt.figure(j+1)
#    plt.plot(np.arange(S_strt,len(S_true_78)+1,1),np.array(val_ls_tot),'--b*',label='valid,j={j}')
#    plt.plot(np.arange(S_strt,len(S_true_78)+1,1),np.array(trn_ls_tot),'--r*',label='train,j={j}')
#    plt.xlabel('Sparsity')    
#    plt.ylabel('Relative valdiation error')    
#    plt.savefig(f'{out_dir_ini}/plots/val_ls_vs_S_j={j}.png',dpi=300)
#import pdb; pdb.set_trace()
## testing OMP using parallel Psi calculation:
#=================================================================================
#===============For Nrep replications run OMP parallel============================
#num_workers = 20
###S_true_78 = [0,60,2,1,74,9,51,3145,48,157,12,63,5,85285,21,229,49,726,80,138,4,79,152,
###                    2664,7,70,2984,2763,9229,714,2754,1449,305,215,11,13,1479,36,2690,218]        
##S_true_78 =[0,1,2,9,60,74,12,28,48,67,215,17,57,41,70,2970]
##S_true_78 =[0,60,2,1,74,9,51,3145] #top_8 coefficients
##S_true_78 =[  0,   1,   2,   3,   4,   7,  10,  16,  17,  21,  82,  99, 142,232]
##S_true_78 = [  0,   2, 1,  3,21,4,82,99,16, 7,  10,  17, 142,232]#[ 0, 21,  2,  4,  3, 82, 99, 45, 16,  1]--top-10 21d problem.
##S_true_78 = [0, 21, 2, 4, 3, 2023, 82, 99, 45, 16]
#S_true_78 = [0,2,3,4,7,10,15,17,18,21,225,2023]
##S_true_78 = [0, 21,2] #, 9, 60,74]
##S_true_78 = [0, 2, 4, 15, 21, 43, 45, 62, 252,484, 486, 503]
##S_true_78 = [0,3,7,60]
##S_true_78 = S_true_78[:6]
#P_0 = int(np.math.factorial(d+p_0)/(np.math.factorial(d)*np.math.factorial(p_0))) 
#Lam_chs_rng = np.setdiff1d(np.arange(0,P_0),np.array(S_true_78))
##rnd_lst = random.sample(Lam_chs_rng.tolist(),2)
##rnd_lst =[18,225] #[80,228] #[18,225]
##S_true_78 = S_true_78 +  rnd_lst
##import pdb; pdb.set_trace()
#S_strt = len(S_true_78) #use this for just one sparsity value 
#chc_lo = 'LS' #'LS' #SO
#part_omp_func = partial(omu.omp_err_for_var_S,S_true_78,S_strt,out_dir_ini,indices0,d,p,y_data,u_data,chc_Psi,chc_omp_slv,Nv,chc_lo,mi_mat,N)
#pool = multiprocessing.Pool(processes=num_workers)
#result_omp_prl = pool.map(part_omp_func,list(j_rng))
#S_list_socv = [result_omp_prl[i_ind]['S'] for i_ind in range(Nrep)]
#pool.close()
#pool.join()    
#import pdb; pdb.set_trace()
#=================================================================================
#=================================================================================
num_workers = args.num_work
part_main_func = partial(mmf.mo_main_utils_function_prll,data_all,out_dir_ini,
                        opt_params,nn_prms_dict,indices0,args,eps_u,W_fac,eps_abs)
pool = multiprocessing.Pool(processes=num_workers)
result_main_prl = pool.map(part_main_func,list(j_rng))
pool.close()
pool.join()    
import pdb; pdb.set_trace()

#plt.show()
end_time = time.time()
print('end - start times:',end_time-start_time)
#print("omph time:",omph_time_end-omph_time_strt)
#print("mo time:",end_time-mo_time_strt)



#

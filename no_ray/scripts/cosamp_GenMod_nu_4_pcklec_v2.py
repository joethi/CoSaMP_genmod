# -*-0 coding: utf-8 -*-
"""
Created on Sat Oct  8 16:30:20 2022
@author: jothi
"""
# coding: utf-8
#=======================================================================================================================================
# This script contains a deep learning based algorithm developed during my master's.
# The name of the thesis is "Neural networks based CoSaMP algorithm for Polynomial Chaos Expansions". 
# The algorithm is called as "Modified Orthogonal Matching Pursuit (OMP)" is implemented in this script. 
# The applications of the algorithm are related to NASA's space missions.
#=======================================================================================================================================
# Import the necessary packages:
#=======================================================================================================================================
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
from ray.tune.schedulers import AsyncHyperBandScheduler 
from ray import tune, air
import multiprocessing
import ray
import argparse
import os
np.random.seed(1)
sys.path.append('/home/jothi/CoSaMP_genNN/no_ray')
sys.path.append('/home/jothi/CoSaMP_genNN/no_ray/scripts/GenMod-org-Hmt')
import genmod_mod_test.polynomial_chaos_utils as pcu
import genmod_mod_test.Gmodel_NN as gnn
import genmod_mod_test.train_NN_omp_wptmg_test as tnn
import genmod_mod_test.omp_utils as omu
import genmod_mod_test.test_coeffs_val_er_utils as tcu
import genmod_mod_test.mo_main_fn_trn3rd_gmd as mmf
import warnings
#=======================================================================================================================================
# Adding Command-Line Argument Parsing to Replace Hardcoded Inputs:
#=======================================================================================================================================
parser = argparse.ArgumentParser(description='Parse the inputs such as output directory name, number of neurons, and so on...')
parser.add_argument('--out_dir_ini',dest = 'out_dir_prs', type=str,\
                        help='specify out_dir_ini directory-it is the location of outputs similar to f\'../output/titan_ppr/results\'')
parser.add_argument('--S0',dest='S_0',default=9,type=int,help='Sparsity for p0')
parser.add_argument('--Sh',dest='S_h',default=7,type=int,help='Sparsity for ph')
parser.add_argument('--N',dest='N_smp',default=100,type=int,help='Number of samples')
parser.add_argument('--Nv',dest='N_v',default=4000,type=int,help='Number of validation samples')
parser.add_argument('--Nt',dest='N_t',default=5,type=int,help='Number of total iterations')
parser.add_argument('--Nrep',dest='N_rep',default=1,type=int,help='Number of sample replications')
parser.add_argument('--Nlhid',dest='N_hid',default=1,type=int,help='Number of sample replications')
parser.add_argument('--hbd',dest='h_bnd',default=7,type=int,help='Number of sample replications')
parser.add_argument('--lr',dest='l_r',default=0.001,type=float,help='Number of sample replications')
parser.add_argument('--pd',dest='p_d',default=0,type=float,help='Number of sample replications')
parser.add_argument('--epochs',dest='ep',default=10000,type=int,help='Number of epochs')
parser.add_argument('--iter_fix',dest='it_fix',default=10000,type=int,help='After this number the metric is evaluated')
parser.add_argument('--tfr',dest='fr_hist',default=10000,type=int,help='d')
parser.add_argument('--nproc',dest='num_work',default=1,type=int,help='After this number the metric is evaluated')
parser.add_argument('--sd_j',dest='sd_indic_ind',default=1,type=int,help='After this number the metric is evaluated')
parser.add_argument('--p_h',dest='ph',default=3,type=int,help='p_h')
parser.add_argument('--case',dest='case_ppr',default='ttn_78',type=str,help='ttn_78, ttn_21, 1dell_14')
parser.add_argument('--p_0',dest='p0',default=2,type=int,help='p_0')
parser.add_argument('--top_i1',dest='topi1',default=3,type=int,help='p_0')
parser.add_argument('--top_i0',dest='topi0',default=3,type=int,help='p_0')
parser.add_argument('--vlcf_add',dest='vlcfadd',default=0,type=int,help='p_0')
parser.add_argument('--d',dest='dim',default=21,type=int,help='d')
parser.add_argument('--dbgit2',dest='dbg_it2',default=0,type=int,help='change after the second iteration')
parser.add_argument('--S_chs',dest='chs_sprs',default=0,type=int,help='S_chs')
parser.add_argument('--S_res',dest='sel_res',default=2,type=int,help='number to include top residual second order coefficients')
parser.add_argument('--S_thrd',dest='sel_thrd',default=2,type=int,help='number to include top residual second order coefficients')
parser.add_argument('--nfld',dest='Nfld_ls',default=5,type=int,help='d')
parser.add_argument('--nfld_t',dest='Nfld_trn',default=5,type=int,help='d')
parser.add_argument('--sdcvls',dest='rnd_st_cvls',default=1,type=int,help='d')
parser.add_argument('--sdcvt',dest='rnd_st_cvtrn',default=1,type=int,help='d')
parser.add_argument('--lscv',dest='ls_cv',default=0,type=int,help='d')
parser.add_argument('--res_tol',dest='resomp_tol',default=1e-10,type=float,help='p_h')
parser.add_argument('--ompsol',dest='chc_omp',default='stdomp',type=str,help='p_h')
parser.add_argument('--poly',dest='chc_poly',default='Hermite',type=str,help='p_h')
parser.add_argument('--qoi',dest='QoI',default='heat_flux',type=str,help='quantity of Interest')
parser.add_argument('--use_gmd',dest='use_gmd',default=0,type=int,help='1-use genmod coefficients for it=0, 2-to use it for it>0')
parser.add_argument('--omponly',dest='omp_only',default=0,type=int,help='switch to use only OMP calculations')
parser.add_argument('--so_res',dest='add_tpso_res',default=0,type=int,\
                        help='0--select 3rd order and residual coeffs for active set,\
                        1--add such that the maximum selection is 2S (previous approch in residual concept),\
                        2--the simple COSAMP approach,3--to use the v3 OMP-select 2 residual 2nd order \
                        coefficient and 2 third order coefficients at most-2S on first iteration,\
                        4--for v4--add 2 residual coefficient to the active set and S-2 top G.')
parser.add_argument('--cini_fl',dest='cht_ini_fl',default='/home/jothi/CoSaMP_genNN/output/titan_ppr/results/\
                        d78_ppr/ref_dbg/cnst_lss_dbg/c_hat_tot_1dellps_n=125_genmod_S=12_1_j0_c0.csv',type=str,\
                        help='file name with the path for initial omp coefficients for reproducing/debugging, use dbg-1 for using this')
parser.add_argument('--ntrial',dest='num_trl',default=10,type=int,help='num_trials')
parser.add_argument('--S_fac',dest='mul_fac',default=2,type=int,help='1-flag for switching to debugging')
parser.add_argument('--dbg',dest='debug_alg',default=0,type=int,help='1-flag for switching to debugging')
parser.add_argument('--pltj',dest='plt_spcdt',default=-1,type=int,help='>0-flag for switching to any other sample')
parser.add_argument('--dbg_act',dest='debug_act',default=2,type=int,help='1-flag for Relu alone,2-flag for None alone ')
parser.add_argument('--tind',dest='dbg_rdtvind',default=0,type=int,help='1-flag for reading train/valid indices from file')
parser.add_argument('--j_rng',dest='j_flg',nargs='+',default=0,type=int,help='0 if all the repications needed, a list having\
                        necessary replication numbers otherwise')
parser.add_argument('--plot_dat',dest='plot_dat',default=0,type=int,help='plot u data?')
parser.add_argument('--ts',dest='tune_sig',default=1,type=int,help='tune signal-0 is for debugging single layer network-errors\
                        out when you use 2 layers and ts 0 concurrently')
args = parser.parse_args()
#=======================================================================================================================================
# Note: Do not use this code for Ncrp>1.
#=======================================================================================================================================
# Set parameters
#=======================================================================================================================================
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
Nlhid = args.N_hid
hid_layers = [args.h_bnd]*Nlhid
GNNmod_ini = gnn.GenNN([d] + hid_layers +[1])
z_n = sum(prm_NN.numel() for prm_NN in GNNmod_ini.state_dict().values())
out_dir_ini = args.out_dir_prs
start_time = time.time()
print('start and start time:',start_time,start_time)
if os.path.exists(f'{out_dir_ini}'):
    print(f"{out_dir_ini} already exists- Do you want to remove directory (Y/n)")
    Answer = input()
    if Answer=="Y":
        os.system(f'rm -r {out_dir_ini}')
    else:
        print("Directory exists already-exiting to prevent overwriting---")
        sys.exit()
os.makedirs(f'{out_dir_ini}')
os.makedirs(f'{out_dir_ini}/plots')
#%% Load data
#=======================================================================================================================================
c_ref = []
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
elif args.case_ppr=="1dell_14":
    chc_Psi = 'Legendre' #'Hermite'
    data = sio.loadmat('../data/dataset1/1Dellptic_data.mat')
    y_data = data['y']
    u_data = data['u'].flatten()
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
elif args.case_ppr=="ttn_20_us3d":
    # Read for the Maximum CN concentration for the previous data:
    out_dir_sct = '../data/Titn_rcnt_dta/d20/d20_ross_sensitivity/Qrad'    
    #y_data1 = pd.read_csv(f'{out_dir_sct}/files/Aval_zis_frm_invmp_78d_N20k.csv').to_numpy() #incorrect file
    y_data1 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N1000_d20.csv').to_numpy()
    y_data = y_data1[:200,0:d] 
    x_data = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_US3D_to_MURP_N200.csv').to_numpy()
    u_data1 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_US3D_to_MURP_N200.csv').to_numpy()
    u_data = np.mean(u_data1,axis=0)    
elif args.case_ppr=="ttn_lwst":
    # Read for the Maximum CN concentration for the previous data:
    #Maximum CN concentration: 
    out_dir_sct = f'../data/Titn_rcnt_dta/d{d}/N=10k' 
    #y_data1 = pd.read_csv(f'{out_dir_sct}/files/Aval_zis_frm_invmp_78d_N20k.csv').to_numpy() #incorrect file
    y_data1 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N10000_d{d}.csv').to_numpy()
    y_data = y_data1[:,0:d] 
    u_data1 = pd.read_csv(f'{out_dir_sct}/xCN_smp.csv').to_numpy()
    u_data = np.amax(u_data1,axis=0)
    print('u_data1[:5,:5]',u_data1[:5,:5])
    print('y_data[:5,:5]',y_data[:5,:5])
    print('shape of u_data1:',u_data1.shape)
    import pdb; pdb.set_trace()
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
    #===========================================================================
    x_data11 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000.csv').to_numpy()
    x_data12 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000_ext1k.csv').to_numpy()
    x_data13 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_Plato_to_MURP_N6000.csv').to_numpy()
    x_data14 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_Plato_to_MURP_N6000_ext6k_4kp2k.csv').to_numpy()
    #x_data = np.hstack((x_data11,x_data12,x_data13)) #FIXME
    x_data = np.hstack((x_data11,x_data12,x_data13,x_data14))
    #===========================================================================
    A_data11 = pd.read_csv(f'{out_dir_sct}/files/Aval_fl_4plto_LN_N1000_d21_sd100_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data12 = pd.read_csv(f'{out_dir_sct}/files/Aval_fl_4plto_LN_N1000_d21_sd200_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data13 = pd.read_csv(f'{out_dir_sct}/files/Aval_fl_4plto_LN_N8000_d21_sd300_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data13 = A_data13[:6000,:d] #FIXME
    A_data14 = pd.read_csv(f'{out_dir_sct}/files/Aval_fl_4plto_LN_N4000_d21_sd400_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data15 = pd.read_csv(f'{out_dir_sct}/files/Aval_fl_4plto_LN_N2000_d21_sd500_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    A_data = np.vstack((A_data11,A_data12,A_data13,A_data14,A_data15))
    #===========================================================================
    if args.QoI!='heat_flux':
        print("Maximum CN concentration is used as QoI")    
        u_data13 = pd.read_csv(f'{out_dir_sct}/xCN_smp_N8k.csv').to_numpy()
        u_data14 = pd.read_csv(f'{out_dir_sct}/xCN_smp_N4k.csv').to_numpy()
        u_data15 = pd.read_csv(f'{out_dir_sct}/xCN_smp_N2k.csv').to_numpy()
        u_data1 = np.hstack((u_data13,u_data14,u_data15))
        u_data = np.amax(u_data1,axis=0)
        #u_data1 = pd.read_csv(f'{out_dir_sct}/xCN_smp_N8k.csv').to_numpy()
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
    ##===========================================================================
    u_data11 = pd.read_csv(f'{out_dir_sct}/d21_dbg/MURP_data/Qrad_dat_Plato_to_MURP_N12000.csv').to_numpy()
    u_data12 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N1000_ext1k.csv').to_numpy()
    u_data13 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N6000.csv').to_numpy()
    u_data14 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N6000_ext6k_4kp2k.csv').to_numpy()
    u_data1 = np.hstack((u_data11,u_data12,u_data13,u_data14))
    u_data = np.mean(u_data1,axis=0)
    ##===========================================================================    
    x_data11 = pd.read_csv(f'{out_dir_sct}/d21_dbg/MURP_data/x_dat_Plato_to_MURP_N12000.csv').to_numpy()
    x_data12 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000_ext1k.csv').to_numpy()
    x_data13 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_Plato_to_MURP_N6000.csv').to_numpy()
    x_data14 = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_Plato_to_MURP_N6000_ext6k_4kp2k.csv').to_numpy()
    #x_data = np.hstack((x_data11,x_data12,x_data13)) #FIXME
    x_data = np.hstack((x_data11,x_data12,x_data13,x_data14))
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
    #Maximum CN concentration:
    out_dir_sct = '../data/Titn_rcnt_dta/d21_dbg'
    y_data1 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N12000_d21_sd100_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
    y_data1[:,d-1] = norm.ppf(0.5*(y_data1[:,d-1]+1))
    y_data = y_data1[:,0:d]
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
elif args.case_ppr=="d20_cnvt":  
    chc_Psi = 'Hermite' #'Hermite'
    out_dir_sct = '../data/Titn_rcnt_dta/d20/d20_ross_sensitivity'
    y_data1 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N1000_d20.csv').to_numpy()
    udata_path = f'{out_dir_sct}/all_qw_stag.3.dat'
    with open(udata_path, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()
        u_dta_lst  = [float(line) for line in lines]
    u_data = np.array(u_dta_lst)
    y_data = y_data1[:200,0:d]
    print("Choice of polynomial required: Hermite")    
    print(f"Choice of polynomial used: {chc_Psi}")    
    print('u_data[:5]',u_data[:5])
    print('y_data[:5,:5]',y_data[:5,:5])    
print('shape of u_data:',u_data.shape)
print('shape of y_data:',y_data.shape)
#=======================================================================================================================================
#%%
Nsmp_tot = np.size(u_data)
#%% Visualize udata:
#=======================================================================================================================================
if pltdta:
    plt.figure(1)
    u_plt_rnd = random.sample(range(2000),200) #FIXME.
    #import pdb; pdb.set_trace()
    for i in u_plt_rnd:
            plt.plot(u_data1[:,i])
            plt.xlabel('spatial location x')
            plt.ylabel('Radiative heat flux[W/cm^2]')
    plt.savefig(f"{out_dir_ini}/u_data.png",dpi=300)
    #%% Visualize ydata:
    # Plot scatter plot of input parameters (Jacqui)
    plt.figure(2)
    y_df = pd.DataFrame(y_data)
    y_df.shape
    sns.pairplot(y_df.iloc[:, :d], kind="scatter")
    plt.savefig(f"{out_dir_ini}/pair_plot_y.png") #,dpi=300)
#%% initial parameters:
mi_mat = pcu.make_mi_mat(d, p)
mi_mat_p0 = pcu.make_mi_mat(d, p_0)
df_mimat = pd.DataFrame(mi_mat)
df_mimat.to_csv(f'{out_dir_ini}/mi_mat_pd={p,d}.csv',index=False)
P = np.size(mi_mat,0)  
data_all = {'y_data':y_data,'u_data':u_data,'mi_mat':mi_mat} 
learning_rate = args.l_r
epochs = args.ep
avtnlst = ['None']*Nlhid + ['expdec']#tune.choice(['None',nn.Sigmoid(),nn.ReLU()])] #[nn.Sigmoid()] # for the final layer by default exp decay is enforced, so the size is number of layers-1.
print('activation used:',avtnlst)
#=======================================================================================================================================
# Write index file:
#=======================================================================================================================================
N = args.N_smp  # note that you need more samples than sparsity for least squares.
Nv = args.N_v
Nrep = args.N_rep
j_rng = range(Nrep) if args.j_flg==0 else args.j_flg #range(Nrep) ---change this to run for a particular replication. Useful for debugging.
print('N:',N,'Nv:',Nv)
#% Save parameters:
opt_params = {'ph':p,'p0':p_0,'d':d,'P':P,'epochs':epochs,'lr':learning_rate,'Sh':S_omp,'S0':S_omp0, 'sprsty':sprsty,
        'N_t':tot_itr,'fr':freq,'W_fac':f'{W_fac}','z_n':z_n,'Tp_i1':top_i1,'Tp_i0':top_i0,'N':N,'Nv':Nv,'Nrep':Nrep,
        'Nc_rp':Nc_rp,'S_chs':S_chs,'chc_poly':chc_Psi,'sd_ind':seed_ind,'sd_thtini':seed_thtini,'sd_ceff':seed_ceff,
        'Nrp_vl':Nrp_vl,"sd_thtini_2nd":sd_thtini_2nd,'iter_fix':args.it_fix,'ntrial':args.num_trl,'Nlhid':Nlhid,
        'chc_omp_slv':chc_omp_slv,'chc_eps':chc_eps}
#import pdb;pdb.set_trace() 
df_params = pd.DataFrame(opt_params,index=[0])
df_params.to_csv(f'{out_dir_ini}/plots/params_genmod_omp_N={N}_ini.csv')
print(df_params)
f = open(f'{out_dir_ini}/plots/1dellps_indices_n={N}_ini.csv', 'w')
N_rndsmp = np.size(u_data)
fw = csv.writer(f)
#header =[*["optim"] * (int(N * 4 / 5)), *["optim"] * (int(N / 5))]
header =[*["optim"] * int(N)]
np.size(header)
fw.writerow(header)
#random.seed(seed_ind) # set seeding for reproducibility/debugging purposes#FIXME seed place..
for i in range(Nrep):
 fw.writerow(random.sample(range(N_rndsmp), N))
f.close()
# FIXME: HARDCODING:
index_file = f'{out_dir_ini}/plots/1dellps_indices_n={N}_ini.csv'
indices0 = pd.read_csv(index_file)
df_indc0 = pd.DataFrame(indices0)
df_indc0.to_csv(f'{out_dir_ini}/plots/indices_genmod_omp_N={N}.csv',index=False)
print('shape of indices0:',indices0.shape)
#=======================================================================================================================================
# Some more parameters:
#=======================================================================================================================================
epsc_omph = []
eps_c_omp_abs = []
eps_abs = []
eps_c = np.zeros(tot_itr+1)
eps_u = np.zeros(tot_itr+1)
nn_prms_dict = {'avtnlst':avtnlst,'hid_layers':hid_layers,'tune_sg':tune_sg}
if args.plt_spcdt>=0:
    j_plt =args.plt_spcdt    
    optim_ind = indices0.iloc[j_plt].to_numpy()
    plt.figure(4)
    plt.plot(np.arange(1,N+1),u_data[optim_ind])
    plt.xlabel('sample index')
    plt.ylabel('Mean radiative heat flux[W/cm^2]')
    plt.savefig(f"{out_dir_ini}/u_data_mean_j{j_plt}.png",dpi=300)
 
print('CAUTION: Do not use it for the training that have more than two checkpoints--\
      will create issues in getting right parameters')
#=======================================================================================================================================
# Run the main module that contains the MO algorithm: 
#=======================================================================================================================================
for j in j_rng:
    mmf.mo_main_utils_function_prll(data_all,out_dir_ini,opt_params,nn_prms_dict,indices0,args,eps_u,W_fac,eps_abs,j)
end_time = time.time()
print('end - start times:',end_time-start_time)
#=======================================================================================================================================

import cProfile
import scipy.io as sio
import seaborn as sns
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import csv
import random
import argparse
from scipy.stats import norm
import time
import os
# np.random.seed(2)
## To call the module in different file. 
import sys
sys.path.append('/home/jothi/CoSaMP_genNN/scripts/GenMod-org-Hmt')
# out_dir_ini = '../output/ttn_chm_ppr/anlrvw23'
import genmod.run_optimizations as ro
import genmod.polynomial_chaos_utils as pcu
seed_ind = 1
#import run_optimizations as ro
#import polynomial_chaos_utils as pcu
st = time.time()
#%%
# data = sio.loadmat(r'C:\Users\jothi\OneDrive - UCB-O365\PhD\UQ_research\ACCESS_UQ\GenMod-NN\GenMod-org\data\Titn_rcnt_dta\Jul28_data_d16.mat')

# y_data = data['Y']
# u_data1 = data['U']

# # u_data1 = np.transpose(u_data2)
# u_data = np.amax(u_data1,axis=1)
# data_all = {'y_data':y_data,'u_data':u_data}
#==============================================================================
parser = argparse.ArgumentParser(description='Parse the inputs such as output directory name, number of neurons, and so on...')
parser.add_argument('--out_dir_ini',dest = 'out_dir_prs',default='../../../output/titan_ppr/results/expalg', type=str, help='specify out_dir_ini directory-it is the location of outputs similar to f\'../output/titan_ppr/results\'')
parser.add_argument('--N',dest='N_smp',default=100,type=int,help='Number of samples')
parser.add_argument('--Nv',dest='N_v',default=4000,type=int,help='Number of validation samples')
parser.add_argument('--n_runs',dest='n_rep',default=1,type=int,help='Number of sample replications')
args = parser.parse_args()
# Set parameters
p = 3
# d = np.size(y_data, 1)
d = 78  # Set smaller value of d for code to run faster
nz = 2*d+1
Chc_Psi = 'Hermite'
mi_mat = pcu.make_mi_mat(d, p)
#==============================================================================
out_dir_ini = args.out_dir_prs
#os.makedirs(f'{out_dir_ini}')
out_dir_sct = '../../../data/Titn_rcnt_dta/dx1em3/LN_d78_hghunkF'
#y_data1 = pd.read_csv(f'{out_dir_sct}/files/Aval_zis_frm_invmp_78d_N20k.csv').to_numpy() #incorrect file
y_data1 = pd.read_csv(f'{out_dir_sct}/files/Aval_zis_frm_invmp_78_ln_hghunkF.csv').to_numpy()
y_data = y_data1[:,0:d]
u_data2 = pd.read_csv(f'{out_dir_sct}/xCN/xCN_smp.csv').to_numpy()
u_data1 = np.transpose(u_data2[:,2:]) #First two colums are just junks.
u_data = np.amax(u_data1,axis=1)
print('u_data1[:5,:5]',u_data1[:5,:5])
print('shape of u_data1:',u_data1.shape)
#y_data11 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N1000_d21_sd100_plsU4_CH4.csv').to_numpy()
#y_data12 = pd.read_csv(f'{out_dir_sct}/files/ynrm_fl_stdGsn_N1000_d21_sd200_plsU4_xCH4_lw0.008_hgh0.02_tht0.004.csv').to_numpy()
#y_data1 = np.vstack((y_data11,y_data12))
##import pdb; pdb.set_trace()
#y_data1[:,d-1] = norm.ppf(0.5*(y_data1[:,d-1]+1))
#u_data11 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_PLATO_to_MURP_N1000.csv').to_numpy()
#u_data12 = pd.read_csv(f'{out_dir_sct}/MURP_data/Qrad_dat_Plato_to_MURP_N1000_ext1k.csv').to_numpy()
#u_data1 = np.hstack((u_data11,u_data12))
#x_data = pd.read_csv(f'{out_dir_sct}/MURP_data/x_dat_PLATO_to_MURP_N1000.csv').to_numpy()
#u_data = np.mean(u_data1,axis=0)
#y_data = y_data1[:,0:d]
data_all = {'y_data':y_data,'u_data':u_data}
plt.figure(21)
#u_plt_rnd = random.sample(range(0,np.size(u_data)), 20)
#u_plt_rnd = random.sample(range(2000,np.size(u_data)), N_y-2000) #FIXME.
u_plt_rnd = random.sample(range(2000),2000) #FIXME.
#import pdb; pdb.set_trace()
#for i in u_plt_rnd:
#    plt.plot(x_data[:,i],u_data1[:,i])
#    # plt.plot(u_data1[i,:],label=f'i={i}')
#    #plt.plot(u_data1[:,i],label=f'i={i}')
#    plt.xlabel('spatial location x')
#    plt.ylabel('Radiative heat flux[W/cm^2]')
#plt.savefig(f"{out_dir_ini}/u_data.png")
##==============================================================================
## Plot scatter plot of input parameters
#plt.figure(1)
#y_df = pd.DataFrame(y_data)
#y_df.shape
#sns.pairplot(y_df.iloc[:, :d], kind="scatter")
#plt.savefig(f"{out_dir_ini}/pair_plot_y.png") #,dpi=300)
#y_df = pd.DataFrame(y_data)
#y_df.shape
#sns.pairplot(y_df.iloc[:, :5], kind="scatter")
#%% Write Psi Matrix:
# df_mi_mat = pd.DataFrame(mi_mat)
# df_mi_mat.to_csv(f'{out_dir_ini}/mi_mat={p,d}.csv',index=False)      
# df_Psi = pd.DataFrame(Psi)
# Psi = pcu.make_Psi(y_data,mi_mat)
# df_Psi.to_csv(f'{out_dir_ini}/Psi_pd={p,d}.csv',index=False)
# # Find least squares coefficients
# Psi_ls = pcu.make_Psi(y_data, mi_mat)
# Psi_ls_T = np.transpose(Psi_ls)
# c_ls = (la.inv(Psi_ls_T @ Psi_ls) @ Psi_ls_T @ u_data).flatten()

# %% # Save sample imdices to file.
# This specifies the indices that will be used for validation and optimization
# for 50 sample replications.
N = args.N_smp
f = open(f'{out_dir_ini}/1dellps_indices_n={N}.csv', 'w')
N_rndsmp = np.size(u_data) #FIXME#np.size(u_data) To generate same results as SO.
random.seed(seed_ind) #setting random seed so that  
fw = csv.writer(f)
header =[*["optim"] * (int(N * 4 / 5)), *["valid"] * (int(N / 5))]
np.size(header)
fw.writerow(header)
for i in range(50):
    fw.writerow(random.sample(range(N_rndsmp), N))
f.close()
# %% Run optimizations
n_runs = args.n_rep  # number of sample replications
strt = 0
n0 = N# number of samples
Nv = args.N_v
gen_params = {'p':p,'d':d,'z_n':nz,'N':N,'Nv':Nv,'Nrep':n_runs,'sd_ind':seed_ind,'Chc_Psi':Chc_Psi}
#import pdb;pdb.set_trace()
df_params = pd.DataFrame(gen_params,index=[0])
df_params.to_csv(f'{out_dir_ini}/params_genmod_omp_N={N}_ini.csv')
print(df_params)

index_file = f'{out_dir_ini}/1dellps_indices_n={n0}.csv'
# index_file = f'{out_dir_ini}/P=3/test/1dellps_indices_n=100_tst1.csv'
# index_file = f'{out_dir_ini}/P=3/test/1dellps_indices_n=100_tst.csv'
# index_file = f'{out_dir_ini}/ini/ind/files/indices_omp_N=300_wrkdomp.csv'
indices0 = pd.read_csv(index_file)
#import pdb; pdb.set_trace()
fn0 = '1dellps_n=' + str(n0)
opt_params = {'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'stepSize': 0.001,
              'maxIter': 100000, 'objecTol': 1e-6, 'ALIter': 10,
              'resultCheckFreq': 10, 'updateLambda': True,
              'switchSigns': False, 'useLCurve': False, 'showLCurve': False,'Nvlrp': 1}
df_opt_params = pd.DataFrame(opt_params,index=[0])
df_opt_params.to_csv(f'{out_dir_ini}/params_genmod_adam_N={N}.csv')
# ro.run_genmod(strt, n_runs - 1, fn0, d, p, data_all, indices0,
#               f'{out_dir_ini}', N, lasso_eps=1e-10,
#               lasso_iter=1e5, lasso_tol=1e-4, lasso_n_alphas=100,
#               opt_params=opt_params)
def prof_rungenmod():
    ro.run_genmod(strt, n_runs - 1, fn0, d, p, data_all, indices0,
              f'{out_dir_ini}', N, Nv,Chc_Psi, mi_mat,nz, lasso_eps=1e-10,
              lasso_iter=1e5, lasso_tol=1e-4, lasso_n_alphas=100,
              opt_params=opt_params)
cProfile.run('prof_rungenmod()','.\profiles\prof.dat')    
# Profiling:
# cProfile.run(ro.run_genmod(strt, n_runs - 1, fn0, d, p, data_all, indices0,
#               'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/ttn_chm_ppr', N, lasso_eps=1e-10,
#               lasso_iter=1e5, lasso_tol=1e-4, lasso_n_alphas=100,
#               opt_params=opt_params))    
# %% Look at results
#vl_err_ac = np.zeros(n_runs)
#for k in range(n_runs):
#    coef_df = pd.read_csv(f'{out_dir_ini}/'+fn0+f'_genmod_kmin=0_{k}.csv')
#    # c = coef_df['GenMod'].to_numpy()
#    c = coef_df['Coefficients'].to_numpy()
#    optim_indices = indices0.iloc[k].to_numpy()
#    valid_indices = np.setdiff1d(range(np.size(u_data)), optim_indices) # on the unseen data.
#
#    valids = [name for name in indices0.columns if name.startswith("valid")]
#    test_indices = indices0.loc[k][valids].to_numpy()  #This is something like a cross validation. part of training. 
#
#    # Testing error
#    #===================================================================
#    # outdir1 = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/1DElliptic/N=4k/genmod_Psi_tot.csv'
#    # Psi_tot = pd.read_csv(outdir1)
#    # op_ind1 = test_indices.tolist()
#    # Psi_test = Psi_tot.loc[op_ind1,:].to_numpy()
#    Psi_test = pcu.make_Psi(y_data[test_indices, :d], mi_mat)
#    test_err = la.norm(
#        Psi_test @ c - u_data[test_indices].T
#    ) / la.norm(u_data[test_indices].T)
#    print(f'Testing Error: {test_err}')
#
#    # Validation error
#    # val_ind1 = valid_indices.tolist()
#    # Psi_valid = Psi_tot.loc[val_ind1[:500],:].to_numpy()
#    Psi_valid = pcu.make_Psi(y_data[valid_indices[:Nv], :d], mi_mat)
#    valid_err = la.norm(
#        Psi_valid @ c - u_data[valid_indices[:Nv]].T
#    ) / la.norm(u_data[valid_indices[:Nv]].T)
#    print(f'Validation Error: {valid_err}')
#    et = time.time()
#    print("Time elapsed (total):",et-st)
#    vl_err_ac[k] = valid_err
#df_vl_err = pd.DataFrame({"vl_err":vl_err_ac})
#df_vl_err.to_csv(f'{out_dir_ini}/vl_err_N{N}.csv',index=False)

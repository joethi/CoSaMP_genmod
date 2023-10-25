# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:41:35 2022

@author: jothi
"""

#%% Least squares:
import csv
import random
import numpy as np
import sklearn.linear_model as lm
import scipy.io as sio
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as la
import time
# np.random.seed(2)
## To call the module in different file. 
import sys
sys.path.append(r'C:\Users\jothi\OneDrive - UCB-O365\PhD\UQ_research\ACCESS_UQ\GenMod-NN\GenMod-org')
out_dir_ini = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/ttn_chm_ppr/P=3/lstsqr'
import genmod.polynomial_chaos_utils as pcu
#import run_optimizations as ro
#import polynomial_chaos_utils as pcu
st = time.time()
#%%
data = sio.loadmat(r'C:\Users\jothi\OneDrive - UCB-O365\PhD\UQ_research\ACCESS_UQ\GenMod-NN\GenMod-org\data\Titn_rcnt_dta\Jul28_data_d16.mat')
# data = sio.loadmat(r'C:\Users\jothi\OneDrive - UCB-O365\PhD\UQ_research\ACCESS_UQ\GenMod-NN\GenMod-org\data\Titn_rcnt_dta\Jul20_data.mat')
y_data = data['Y']
u_data1 = data['U']
# u_data1 = np.transpose(u_data2)
u_data = np.amax(u_data1,axis=1)
data_all = {'y_data':y_data,'u_data':u_data}
#===================================================
# Plot scatter plot of input parameters
y_df = pd.DataFrame(y_data)
y_df.shape
sns.pairplot(y_df.iloc[:, :5], kind="scatter")
plt.show()

# Set parameters:
p = 3
# d = np.size(y_data, 1)
d = 16  # Set smaller value of d for code to run faster
nz = 2*d+1
n_runs = 10
N = 9000
N_te = 1000
mi_mat = pcu.make_mi_mat(d, p)
#%% Write Psi Matrix:
# Psi = pd.read_csv(f'{out_dir_ini}/Psi_pd=(2, 78).csv').to_numpy()    
Psi = pcu.make_Psi(y_data,mi_mat)
df_Psi = pd.DataFrame(Psi)
df_Psi.to_csv(f'{out_dir_ini}/Psi_pd={p,d}.csv',index=False)
# Psi_ls = Psi[:N,:]
# Psi_ls = pcu.make_Psi(y_data,mi_mat)
# Psi_ls = np.array([[1,2,3],[4,5,6]]) 
# P = np.size(Psi_ls,1)
# # Find least squares coefficients
#%% validation:   
f = open(f'{out_dir_ini}/1dellps_indices_n={N}.csv', 'w')
N_tot = np.size(u_data)
fw = csv.writer(f)
header =[*["optim"] * (int(N * 4 / 5)), *["optim"] * (int(N / 5))]
np.size(header)
fw.writerow(header)
for i in range(n_runs):
    fw.writerow(random.sample(range(N_tot), N))
f.close()
index_file = f'{out_dir_ini}/1dellps_indices_n={N}.csv'
# index_file = f'{out_dir_ini}/P=3/test/1dellps_indices_n=100_tst1.csv'
indices0 = pd.read_csv(index_file)
eps_u=np.zeros(n_runs)
for i in range(n_runs):
    optim_indices = indices0.iloc[i].to_numpy()
    test_indices = np.setdiff1d(range(np.size(u_data)), optim_indices)
    Psi_ls = Psi[optim_indices,:]
    Psi_ls_T = np.transpose(Psi_ls)
    c_ls = (la.inv(Psi_ls_T @ Psi_ls) @ Psi_ls_T @ u_data[optim_indices]).flatten()
    eps_u[i] = la.norm(Psi[test_indices[:N_te],:]@c_ls-u_data[test_indices[:N_te]].T)/la.norm(u_data[test_indices[:N_te]].T)
#%% Load to file:
df_eps_u = pd.DataFrame(eps_u)
df_eps_u.to_csv(f'{out_dir_ini}/eps_u_lsqr_pd={(p,d)}_{N}_Nte={N_te}.csv')       
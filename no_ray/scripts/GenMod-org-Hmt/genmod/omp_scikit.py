# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:18:06 2022

@author: jothi
"""
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
out_dir_ini = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/ttn_chm_ppr/P=3/omp'
import genmod.polynomial_chaos_utils as pcu
#import run_optimizations as ro
#import polynomial_chaos_utils as pcu
st = time.time()
#%%
data = sio.loadmat(r'C:\Users\jothi\OneDrive - UCB-O365\PhD\UQ_research\ACCESS_UQ\GenMod-NN\GenMod-org\data\Titn_rcnt_dta\Jul28_data_d16.mat')
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

# Set parameters
p = 3
# d = np.size(y_data, 1)
d = 16  # Set smaller value of d for code to run faster
nz = 2*d+1
n_runs = 10
N = 5000
N_te = 1000
mi_mat = pcu.make_mi_mat(d, p)
#%% Write Psi Matrix:
Psi_ls = pd.read_csv(f'{out_dir_ini}/Psi_pd=(3, 16).csv').to_numpy()    
# Psi_ls = pcu.make_Psi(y_data,mi_mat)
# Psi_ls = np.array([[1,2,3],[4,5,6]]) 
P = np.size(Psi_ls,1)
# # Find least squares coefficients
# Psi_ls_T = np.transpose(Psi_ls)
# c_ls = (la.inv(Psi_ls_T @ Psi_ls) @ Psi_ls_T @ u_data).flatten()
#%% Find mutual incoherence:
# def corr(x,y):
#     return abs(np.dot(x,y))/(np.sqrt((x**2).sum())*np.sqrt((y**2).sum()))
# ind = np.zeros(P)
# n = 1 
# for i in range(P):
#     y = Psi_ls[:N,i]
#     corrs = [ corr(x,y) for x in Psi_ls[:N,:].T]
#     corrs[i] = 0.0
#     ind[i] = np.amax(corrs)
# corr_mx = np.amax(ind)       
#%% Calculate signs using OMP:    
def omp_c(Psi, u_data, all=True):
    """Set coefficient signs using Orthogonal Matching Pusuit method.""" 
    # P = np.size(Psi, 1)

    # Find initial signs with orthogonal matching pursuit
    omp = lm.OrthogonalMatchingPursuitCV(cv=5, fit_intercept=False)
    omp.fit(Psi, u_data)
    c = omp.coef_
    return c
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
    c_omp = omp_c(Psi_ls[optim_indices,:],u_data[optim_indices])
    eps_u[i] = la.norm(Psi_ls[test_indices[:N_te],:]@c_omp.T-u_data[test_indices[:N_te]].T)/la.norm(u_data[test_indices[:N_te]].T)
#%% error calculation:
# eps_c = la.norm(c_omp-c_ls)/la.norm(c_ls)    
#%% Load to file:
df_eps_u = pd.DataFrame(eps_u)
df_eps_u.to_csv(f'{out_dir_ini}/eps_u_omp_{N}.csv')    
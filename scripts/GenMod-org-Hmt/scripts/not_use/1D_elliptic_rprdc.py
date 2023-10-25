# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:00:07 2022

@author: jothi
"""
# %%
import pickle
import csv
import scipy.io as sio
import pandas as pd
import numpy as np
import sys
sys.path.append('C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org')
out_dir_ini = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/1DElliptic_ppr/P=3'
import genmod.polynomial_chaos_utils as pcu
#%%
data = sio.loadmat('C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/data/dataset1/1Dellptic_data.mat')
y_data = data['y']
u_data = data['u']
p = 2
d = 14
mi_mat = pcu.make_mi_mat(d,p)
Psi = pcu.make_Psi(y_data,mi_mat)
df_Psi = pd.DataFrame(Psi)
df_Psi.to_csv(f'{out_dir_ini}/Psi_pd={p,d}.csv',index=False)
# %%
# nsamp = 50
# Nset = [320]
# Ns_len = len(Nset)
# # n = 25
# dmo_err = np.zeros((nsamp,Ns_len))
# las_err = np.zeros((nsamp,Ns_len))
# fn = f'{out_dir_ini}/N={Nset[0]}/files/1dellps_n={Nset[0]}'
# dmo = pd.read_csv(fn + '_genmod_' + str(0) + '.csv')
# for n in range(Ns_len):
#     # fn0 = '1dellps_n=' + str(Nset[n])
#     fn = f'{out_dir_ini}/N={Nset[n]}/files/1dellps_n={Nset[n]}'
#     indices1 = pd.read_csv(f'{out_dir_ini}/N={Nset[n]}/files/1dellps_indices_n={Nset[n]}.csv')    
#     for i in range(nsamp):
#         #print(i)
#         # dmo = pickle.load(open(fn + '_' + str(i) + '.pkl','rb'))
#         dmo = pd.read_csv(fn + '_genmod_' + str(i) + '.csv')
#         c_m = dmo['Coefficients'].to_numpy()
#         data_indices = indices1.loc[i].to_numpy()
#         not_data_indices = np.setdiff1d(range(np.size(u_data)),data_indices)
#         dmo_err[i,n] = np.linalg.norm(Psi[not_data_indices]@c_m-u_data[not_data_indices])/np.linalg.norm(u_data[not_data_indices])       
#         # eps_c[i,n] = np.linalg.norm(Psi[not_data_indices]@dmo.c-u_data[not_data_indices])/np.linalg.norm(u_data[not_data_indices])        
#         #print(dmo.z)
#         # opening the CSV file
#         # las = pickle.load(open('/app/paper_optimizations/example3/data3_n=135_wlas3_'+str(i) + '.pkl','rb'))
#         # las_coef = las.coef_
#         # las = pd.read_csv('/app/current_output/data3_n=25_omp_' + str(i)+'.csv')
#         # las_coef = las.iloc[:,0].to_numpy()
#         #print(las_coef)
#         #print(np.linalg.norm(Psi[not_data_indices,:]@dmo.c-u_data[not_data_indices])/np.linalg.norm(u_data[not_data_indices]))
        
#         # las_err[i] = np.linalg.norm(Psi[not_data_indices]@las_coef-u_data[not_data_indices])/np.linalg.norm(u_data[not_data_indices])
#         #print(las_err)
#         #print(dmo_err)
#         #print((las_err-dmo_err)/las_err)
#         #print(Psi[not_data_indices,:]@las_coef-u_data[not_data_indices]-(Psi[not_data_indices]@dmo.c-u_data[not_data_indices]))
# # df_err = pd.DataFrame({'eps_u':dmo_err})
# # df_err.to_csv(f'{out_dir_ini}',index=False) 
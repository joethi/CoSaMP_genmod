import scipy.io as sio
import seaborn as sns
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import csv
import random
import time
# np.random.seed(2)
## To call the module in different file. 
import sys
sys.path.append(r'C:\Users\jothi\OneDrive - UCB-O365\PhD\UQ_research\ACCESS_UQ\GenMod-NN\GenMod-org')
out_dir_ini = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/sqr_cvty_ppr'
import genmod.run_optimizations as ro
import genmod.polynomial_chaos_utils as pcu
#import run_optimizations as ro
#import polynomial_chaos_utils as pcu
st = time.time()
#============================Change here=======================================
# data = sio.loadmat(r'C:\Users\jothi\OneDrive - UCB-O365\PhD\UQ_research\ACCESS_UQ\GenMod-NN\GenMod-NN0nu\output\1DElliptic\N=40k\1Dell_dt_tst.mat')
# data = sio.loadmat(r'C:\Users\jothi\OneDrive - UCB-O365\PhD\UQ_research\ACCESS_UQ\GenMod-NN\GenMod-org\data\dataset1\N=500\1Dell_dt_tst.mat')
# data = sio.loadmat(r'C:\Users\jothi\OneDrive - UCB-O365\PhD\UQ_research\ACCESS_UQ\GenMod-NN\GenMod-org\data\dataset1\1Dellptic_data.mat')
# y_data = data['y']
# u_data = data['u']
#%%
y_dat = sio.loadmat(r'C:\Users\jothi\OneDrive - UCB-O365\PhD\UQ_research\ACCESS_UQ\GenMod-NN\GenMod-org\data\dataset3\xi256.mat')
u_dat = sio.loadmat(r'C:\Users\jothi\OneDrive - UCB-O365\PhD\UQ_research\ACCESS_UQ\GenMod-NN\GenMod-org\data\dataset3\U128.mat')
y_data3_temp = y_dat['xi']
y_data =np.copy(y_data3_temp)

#Fix variables not distributed on [-1,1]
y_data[:,1]=2*(y_data3_temp[:,1]-105)/(np.max(y_data3_temp[:,1])-105)-1
y_data[:,0]=2*(y_data3_temp[:,0]-.004)/(.01-.004)-1

#Verify random variables are contained in [-1,1]
for i in range(np.size(y_data,1)):
    print('Min: ' + str(np.min(y_data[:,i])))
    print('Max: ' + str(np.max(y_data[:,i])))

#Get output data from center of domain (grid w/out boundary is 126, so pick 63)
u_data3_tot = u_dat['U']
u_data = u_data3_tot[62,:]

data_all = {'y_data':y_data,'u_data':u_data}
#===================================================
# Plot scatter plot of input parameters
y_df = pd.DataFrame(y_data)
y_df.shape
sns.pairplot(y_df.iloc[:, :5], kind="scatter")
plt.show()

# Set parameters
p = 2
# d = np.size(y_data, 1)
d = 52  # Set smaller value of d for code to run faster
mi_mat = pcu.make_mi_mat(d, p)
#%% Write Psi Matrix:
Psi = pcu.make_Psi(y_data,mi_mat)
df_Psi = pd.DataFrame(Psi)
df_Psi.to_csv(f'{out_dir_ini}/Psi_pd={p,d}.csv',index=False)
# Find least squares coefficients
Psi_ls = pcu.make_Psi(y_data, mi_mat)
Psi_ls_T = np.transpose(Psi_ls)
c_ls = (la.inv(Psi_ls_T @ Psi_ls) @ Psi_ls_T @ u_data).flatten()

# %% # Save sample imdices to file.
# This specifies the indices that will be used for validation and optimization
# for 50 sample replications.

N = 80
f = open(f'{out_dir_ini}/1dellps_indices_n={N}.csv', 'w')
N_tot = np.size(u_data)
fw = csv.writer(f)
header =[*["optim"] * (int(N * 4 / 5)), *["valid"] * (int(N / 5))]
np.size(header)
fw.writerow(header)
for i in range(50):
    fw.writerow(random.sample(range(N_tot), N))
f.close()
# %% Run optimizations
n_runs = 50  # number of sample replications
strt = 0
n0 = N# number of samples
index_file = f'{out_dir_ini}/1dellps_indices_n={n0}.csv'
# index_file = f'{out_dir_ini}/1dellps_indices_n=40.csv'
# index_file = f'{out_dir_ini}/P=3/N=660/files/1dellps_indices_n=660.csv'
indices0 = pd.read_csv(index_file)
fn0 = '1dellps_n=' + str(n0)
opt_params = {'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'stepSize': 0.001,
              'maxIter': 100000, 'objecTol': 1e-6, 'ALIter': 10,
              'resultCheckFreq': 10, 'updateLambda': True,
              'switchSigns': False, 'useLCurve': False, 'showLCurve': False}
ro.run_genmod(strt, n_runs - 1, fn0, d, p, data_all, indices0,
              'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/sqr_cvty_ppr', lasso_eps=1e-10,
              lasso_iter=1e5, lasso_tol=1e-4, lasso_n_alphas=100,
              opt_params=opt_params)
# %% Look at results

coef_df = pd.read_csv(f'{out_dir_ini}/'+fn0+'_genmod_0.csv')
# c = coef_df['GenMod'].to_numpy()
c = coef_df['Coefficients'].to_numpy()
optim_indices = indices0.iloc[0].to_numpy()
valid_indices = np.setdiff1d(range(np.size(u_data)), optim_indices)

valids = [name for name in indices0.columns if name.startswith("valid")]
test_indices = indices0.loc[0][valids].to_numpy()

# Testing error
#===================================================================
# outdir1 = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/1DElliptic/N=4k/genmod_Psi_tot.csv'
# Psi_tot = pd.read_csv(outdir1)
# op_ind1 = test_indices.tolist()
# Psi_test = Psi_tot.loc[op_ind1,:].to_numpy()
Psi_test = pcu.make_Psi(y_data[test_indices, :d], mi_mat)
test_err = la.norm(
    Psi_test @ c - u_data[test_indices].T
) / la.norm(u_data[test_indices].T)
print(f'Testing Error: {test_err}')

# Validation error
# val_ind1 = valid_indices.tolist()
# Psi_valid = Psi_tot.loc[val_ind1[:500],:].to_numpy()
Psi_valid = pcu.make_Psi(y_data[valid_indices[:500], :d], mi_mat)
valid_err = la.norm(
    Psi_valid @ c - u_data[valid_indices[:500]].T
) / la.norm(u_data[valid_indices[:500]].T)
print(f'Validation Error: {valid_err}')
et = time.time()
print("Time elapsed (total):",et-st)
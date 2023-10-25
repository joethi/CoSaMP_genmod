import cProfile
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
out_dir_ini = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/wing_wght_ppr'
import genmod.run_optimizations as ro
import genmod.polynomial_chaos_utils as pcu
#import run_optimizations as ro
#import polynomial_chaos_utils as pcu
st = time.time()
#%%
data = sio.loadmat(r'C:\Users\jothi\OneDrive - UCB-O365\PhD\UQ_research\ACCESS_UQ\GenMod-NN\GenMod-org\data\wing_weight\wingweight_data.mat')
y_data = data['y']
u_data1 = data['u']
u_data = u_data1.flatten()
# u_data1 = np.transpose(u_data2)
# u_data = np.amax(u_data1,axis=1)
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
d = 10  # Set smaller value of d for code to run faster
nz = 2*d+1
mi_mat = pcu.make_mi_mat(d, p)
#%% Case 1:
N = 40
f = open(f'{out_dir_ini}/1dellps_indices_n={N}.csv', 'w')
N_tot = np.size(u_data)
fw = csv.writer(f)
header =[*["optim"] * (int(N * 4 / 5)), *["valid"] * (int(N / 5))]
np.size(header)
fw.writerow(header)
for i in range(50):
    fw.writerow(random.sample(range(N_tot), N))
f.close()
# Run optimizations
n_runs = 50  # number of sample replications
strt = 0
n0 = N# number of samples
index_file = f'{out_dir_ini}/1dellps_indices_n={n0}.csv'
indices0 = pd.read_csv(index_file)
fn0 = '1dellps_n=' + str(n0)
opt_params = {'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'stepSize': 0.001,
              'maxIter': 100000, 'objecTol': 1e-6, 'ALIter': 10,
              'resultCheckFreq': 10, 'updateLambda': True,
              'switchSigns': False, 'useLCurve': False, 'showLCurve': False,'Nvlrp': 1}
df_opt_params = pd.DataFrame(opt_params,index=[0])
df_opt_params.to_csv(f'{out_dir_ini}/params_genmod_adam_N={N}.csv')
out_dir_ini = f'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/wing_wght_ppr/d=10/p=3/N={N}'
ro.run_genmod(strt, n_runs - 1, fn0, d, p, data_all, indices0,
          f'{out_dir_ini}', N, mi_mat, lasso_eps=1e-10,
          lasso_iter=1e5, lasso_tol=1e-3, lasso_n_alphas=100,
          opt_params=opt_params)
#%% Case 2:
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
# Run optimizations
n_runs = 50  # number of sample replications
strt = 0
n0 = N# number of samples
index_file = f'{out_dir_ini}/1dellps_indices_n={n0}.csv'
indices0 = pd.read_csv(index_file)
fn0 = '1dellps_n=' + str(n0)
out_dir_ini = f'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/wing_wght_ppr/d=10/p=3/N={N}'
ro.run_genmod(strt, n_runs - 1, fn0, d, p, data_all, indices0,
          f'{out_dir_ini}', N, mi_mat, lasso_eps=1e-10,
          lasso_iter=1e5, lasso_tol=1e-3, lasso_n_alphas=100,
          opt_params=opt_params)
#%% Case 3:
N = 160
f = open(f'{out_dir_ini}/1dellps_indices_n={N}.csv', 'w')
N_tot = np.size(u_data)
fw = csv.writer(f)
header =[*["optim"] * (int(N * 4 / 5)), *["valid"] * (int(N / 5))]
np.size(header)
fw.writerow(header)
for i in range(50):
    fw.writerow(random.sample(range(N_tot), N))
f.close()
# Run optimizations
n_runs = 50  # number of sample replications
strt = 0
n0 = N# number of samples
index_file = f'{out_dir_ini}/1dellps_indices_n={n0}.csv'
indices0 = pd.read_csv(index_file)
fn0 = '1dellps_n=' + str(n0)
out_dir_ini = f'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/wing_wght_ppr/d=10/p=3/N={N}'
ro.run_genmod(strt, n_runs - 1, fn0, d, p, data_all, indices0,
          f'{out_dir_ini}', N, mi_mat, lasso_eps=1e-10,
          lasso_iter=1e5, lasso_tol=1e-3, lasso_n_alphas=100,
          opt_params=opt_params)
#%% Case 4:
N = 200
f = open(f'{out_dir_ini}/1dellps_indices_n={N}.csv', 'w')
N_tot = np.size(u_data)
fw = csv.writer(f)
header =[*["optim"] * (int(N * 4 / 5)), *["valid"] * (int(N / 5))]
np.size(header)
fw.writerow(header)
for i in range(50):
    fw.writerow(random.sample(range(N_tot), N))
f.close()
# Run optimizations
n_runs = 50  # number of sample replications
strt = 0
n0 = N# number of samples
index_file = f'{out_dir_ini}/1dellps_indices_n={n0}.csv'
indices0 = pd.read_csv(index_file)
fn0 = '1dellps_n=' + str(n0)
out_dir_ini = f'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/wing_wght_ppr/d=10/p=3/N={N}'
ro.run_genmod(strt, n_runs - 1, fn0, d, p, data_all, indices0,
          f'{out_dir_ini}', N, mi_mat, lasso_eps=1e-10,
          lasso_iter=1e5, lasso_tol=1e-3, lasso_n_alphas=100,
          opt_params=opt_params)
#%% Case 5:
N = 250
f = open(f'{out_dir_ini}/1dellps_indices_n={N}.csv', 'w')
N_tot = np.size(u_data)
fw = csv.writer(f)
header =[*["optim"] * (int(N * 4 / 5)), *["valid"] * (int(N / 5))]
np.size(header)
fw.writerow(header)
for i in range(50):
    fw.writerow(random.sample(range(N_tot), N))
f.close()
# Run optimizations
n_runs = 50  # number of sample replications
strt = 0
n0 = N# number of samples
index_file = f'{out_dir_ini}/1dellps_indices_n={n0}.csv'
indices0 = pd.read_csv(index_file)
fn0 = '1dellps_n=' + str(n0)
out_dir_ini = f'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/wing_wght_ppr/d=10/p=3/N={N}'
ro.run_genmod(strt, n_runs - 1, fn0, d, p, data_all, indices0,
          f'{out_dir_ini}', N, mi_mat, lasso_eps=1e-10,
          lasso_iter=1e5, lasso_tol=1e-3, lasso_n_alphas=100,
          opt_params=opt_params)
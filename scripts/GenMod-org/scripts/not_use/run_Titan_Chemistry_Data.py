import scipy.io as sio
import seaborn as sns
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import csv
import random
## To call the module in different file. 
import sys
sys.path.append(r'C:\Users\jothi\OneDrive - UCB-O365\PhD\UQ_research\ACCESS_UQ\GenMod-main\GenMod-main')
import genmod.run_optimizations as ro
import genmod.polynomial_chaos_utils as pcu
outdir_ini = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-org/output/titan_chemistry_dataset'
#import run_optimizations as ro
#import polynomial_chaos_utils as pcu
data = sio.loadmat(r'C:\Users\jothi\OneDrive - UCB-O365\PhD\UQ_research\ACCESS_UQ\GenMod-NN\GenMod-org\data\Titan_Chemistry_Data\Titan_Chemistry_Data.mat')

y_data = data['y_samp']
u_data = data['u_samp']


# Plot scatter plot of input parameters
y_df = pd.DataFrame(y_data)
y_df.shape
sns.pairplot(y_df.iloc[:, :5], kind="scatter")
plt.show()

# Set parameters
p = 2
# d = np.size(y_data, 1)
d = 20  # Set smaller value of d for code to run faster
data_all = {'y_data': y_data[:, :d], 'u_data': u_data}
mi_mat = pcu.make_mi_mat(d, p)

# # Find least squares coefficients
# Psi_ls = pcu.make_Psi(y_data, mi_mat)
# Psi_ls_T = np.transpose(Psi_ls)
# c_ls = (la.inv(Psi_ls_T @ Psi_ls) @ Psi_ls_T @ u_data).flatten()

# %% # Save sample imdices to file.
# This specifies the indices that will be used for validation and optimization
# for 50 sample replications.

N = 1000
f = open(f'{outdir_ini}/Titan_indices_n={N}.csv', 'w')
N_tot = np.size(u_data)
fw = csv.writer(f)
header =[*["optim"] * (int(N * 4 / 5)), *["valid"] * (int(N / 5))]
np.size(header)
fw.writerow(header)
for i in range(50):
    fw.writerow(random.sample(range(N_tot), N))
f.close()

# %% Run optimizations
n_runs = 1  # number of sample replications
n0 = N# number of samples
index_file = f'{outdir_ini}/Titan_indices_n={n0}.csv'
indices0 = pd.read_csv(index_file)
fn0 = 'Titan_n=' + str(n0)
opt_params = {'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'stepSize': 0.001,
              'maxIter': 100000, 'objecTol': 1e-6, 'ALIter': 10,
              'resultCheckFreq': 10, 'updateLambda': True,
              'switchSigns': False, 'useLCurve': False, 'showLCurve': False}
ro.run_genmod(0, n_runs - 1, fn0, d, p, data_all, indices0,
              f'{outdir_ini}', lasso_eps=1e-10,
              lasso_iter=1e5, lasso_tol=1e-4, lasso_n_alphas=100,
              opt_params=opt_params)

# %% Look at results

coef_df = pd.read_csv(f'{outdir_ini}/Titan_n={N}_genmod_0.csv')
c = coef_df['Coefficients'].to_numpy()
optim_indices = indices0.iloc[0].to_numpy()
valid_indices = np.setdiff1d(range(np.size(u_data)), optim_indices)

valids = [name for name in indices0.columns if name.startswith("valid")]
test_indices = indices0.loc[0][valids].to_numpy()

# Testing error
Psi_test = pcu.make_Psi(y_data[test_indices, :d], mi_mat)
test_err = la.norm(
    Psi_test @ c - u_data[test_indices].T
) / la.norm(u_data[test_indices].T)
print(f'Testing Error: {test_err}')

# Validation error
Psi_valid = pcu.make_Psi(y_data[valid_indices[:500], :d], mi_mat)
valid_err = la.norm(
    Psi_valid @ c - u_data[valid_indices[:500]].T
) / la.norm(u_data[valid_indices[:500]].T)
print(f'Valiidation Error: {valid_err}')
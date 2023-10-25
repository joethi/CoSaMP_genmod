import sys
import scipy.io as sio
import numpy as np
import pandas as pd
import csv
import random

import genmodpce.polynomial_chaos_utils as pcu
import genmodpce.run_optimizations as ro

import importlib


# %%
oldout = sys.stdout
olderr = sys.stderr

# %%
sys.stdout = oldout
sys.stderr = olderr

# %% Dataset 1
y_data1 = np.genfromtxt('/app/data/dataset1/xi_40000_1.dat')
u_data1 = np.genfromtxt('/app/data/dataset1/fsamp_40000_1.dat')
data1_all = {'y_data':y_data1,'u_data':u_data1}

#Verify random variables are contained in [-1,1]
for i in range(np.size(y_data1,1)):
    print('Min: ' + str(np.min(y_data1[:,i])))
    print('Max: ' + str(np.max(y_data1[:,i])))

# #Save sammple imdices to file
# N = 30
# f = open('/app/current_output/example1_indices_n='+str(N)+'.csv', 'w')
# N_tot = np.size(u_data1)
# fw = csv.writer(f)
# header =[*["optim"]*(int(N*4/5)),*["valid"]*(int(N/5))]
# np.size(header)
# fw.writerow(header)
# for i in range(50):
#     fw.writerow(random.sample(range(N_tot),N))
# f.close()

#Set parameters
d = 14
p = 3

#Find least squares coefficients
mi_mat = pcu.make_mi_mat(d,p)
Psi_ls = pcu.make_Psi(y_data1,mi_mat)
Psi_ls_T = np.transpose(Psi_ls)
c_ls = (np.linalg.inv(Psi_ls_T @ Psi_ls) @ Psi_ls_T @ u_data1).flatten()

#Save least squares coefficientsresults
df_ls_coef1 = pd.DataFrame({'Coefficients': c_ls})
df_ls_coef1.to_csv('/app/current_output/data1_ls_coefs_N='+str(np.size(u_data1))+'.csv',index=False)

#Run generative model/sparse optimizations
importlib.reload(ro)
n0 = 30
indices0 = pd.read_csv('/app/paper_optimizations/example1_indices_n='+str(n0)+'.csv')
fn0 = 'data1_n=' + str(n0)
ro.run_l1(0,49,fn0,d,p,data1_all,indices0,lasso_eps=1e-10,lasso_iter=1e5,lasso_tol=1e-4,lasso_n_alphas=100)
ro.run_genmod(0,49,fn0,d,p,data1_all,indices0,lasso_eps=1e-10,lasso_iter=1e5,lasso_tol=1e-4,lasso_n_alphas=100)

n1 = 40
indices1 = pd.read_csv('/app/paper_optimizations/example1_indices_n='+str(n1)+'.csv')
fn1 = 'data1_n=' + str(n1)
ro.run_l1(0,49,fn1,d,p,data1_all,indices1,lasso_eps=1e-10,lasso_iter=1e5,lasso_tol=1e-4,lasso_n_alphas=100)
ro.run_genmod(0,49,fn1,d,p,data1_all,indices1,lasso_eps=1e-10,lasso_iter=1e5,lasso_tol=1e-4,lasso_n_alphas=100)

n2 = 80
indices2 = pd.read_csv('/app/paper_optimizations/example1_indices_n='+str(n2)+'.csv')
fn2 = 'data1_n=' + str(n2)
ro.run_l1(0,49,fn2,d,p,data1_all,indices2,lasso_eps=1e-10,lasso_iter=1e5,lasso_tol=1e-4,lasso_n_alphas=100)
ro.run_genmod(0,49,fn2,d,p,data1_all,indices2,lasso_eps=1e-10,lasso_iter=1e5,lasso_tol=1e-4,lasso_n_alphas=100)

n3 = 160
indices3 = pd.read_csv('/app/paper_optimizations/example1_indices_n='+str(n3)+'.csv')
fn3 = 'data1_n=' + str(n3)
ro.run_l1(0,49,fn3,d,p,data1_all,indices3,lasso_eps=1e-10,lasso_iter=1e5,lasso_tol=1e-4,lasso_n_alphas=100)
ro.run_genmod(0,49,fn3,d,p,data1_all,indices3,lasso_eps=1e-10,lasso_iter=1e5,lasso_tol=1e-4,lasso_n_alphas=100)

n4 = 320
indices4 = pd.read_csv('/app/paper_optimizations/example1_indices_n='+str(n4)+'.csv')
fn4 = 'data1_n=' + str(n4)
ro.run_l1(0,49,fn4,d,p,data1_all,indices4,lasso_eps=1e-10,lasso_iter=1e5,lasso_tol=1e-4,lasso_n_alphas=100)
ro.run_genmod(0,49,fn4,d,p,data1_all,indices4,lasso_eps=1e-10,lasso_iter=1e5,lasso_tol=1e-4,lasso_n_alphas=100)

# %% Dataset 2
y_dat = sio.loadmat('/app/data/dataset2/y_samples.mat')
u_dat = sio.loadmat('/app/data/dataset2/u_samples.mat')
y_data2 = y_dat['y_samples']
u_data2 = u_dat['u_samples'].flatten()
data2_all = {'y_data':y_data2,'u_data':u_data2}

#Verify random variables are contained in [-1,1]
for i in range(np.size(y_data2,1)):
    print('Min: ' + str(np.min(y_data2[:,i])))
    print('Max: ' + str(np.max(y_data2[:,i])))

# #Save sammple imdices to file
# N = 75
# f = open('/app/current_output/example2_indices_n='+str(N)+'.csv', 'w')
# N_tot = np.size(u_data2)
# fw = csv.writer(f)
# header =[*["optim"]*(int(N*4/5)),*["valid"]*(int(N/5))]
# np.size(header)
# fw.writerow(header)
# for i in range(50):
#     fw.writerow(random.sample(range(N_tot),N))
# f.close()

#Set parameters
d = 20
p = 3

#Find least squares coefficients
mi_mat = pcu.make_mi_mat(d,p)
Psi_ls = pcu.make_Psi(y_data2,mi_mat)
Psi_ls_T = np.transpose(Psi_ls)
c_ls = (np.linalg.inv(Psi_ls_T @ Psi_ls) @ Psi_ls_T @ u_data2).flatten()

#Save least squares coefficients results
df_ls_coef2 = pd.DataFrame({'Least_Squares_Coefficients': c_ls})
df_ls_coef2.to_csv('/app/current_output/data2_ls_coefs_N='+str(np.size(u_data2))+'.csv')

### Run generative model optimizations
n1 = 60
indices1 = pd.read_csv('/app/paper_optimizations/example2_indices_n='+str(n1)+'.csv')
fn1 = 'data2_n=' + str(n1)
ro.run_l1(0,49,fn1,d,p,data2_all,indices1,lasso_eps=1e-10,lasso_iter=1e5,lasso_tol=1e-4,lasso_n_alphas=100)

importlib.reload(ro)
ro.run_genmod(28,49,fn1,d,p,data2_all,indices1,lasso_eps=1e-10,lasso_iter=1e5,lasso_tol=1e-4,lasso_n_alphas=100)

# %% Dataset 3

y_dat = sio.loadmat('/app/data/dataset3/xi256.mat')
u_dat = sio.loadmat('/app/data/dataset3/U128.mat')
y_data3_temp = y_dat['xi']
y_data3 =np.copy(y_data3_temp)

#Fix variables not distributed on [-1,1]
y_data3[:,1]=2*(y_data3_temp[:,1]-105)/(np.max(y_data3_temp[:,1])-105)-1
y_data3[:,0]=2*(y_data3_temp[:,0]-.004)/(.01-.004)-1

#Verify random variables are contained in [-1,1]
for i in range(np.size(y_data3,1)):
    print('Min: ' + str(np.min(y_data3[:,i])))
    print('Max: ' + str(np.max(y_data3[:,i])))

#Get output data from center of domain (grid w/out boundary is 126, so pick 63)
u_data3_tot = u_dat['U']
u_data3 = u_data3_tot[62,:]

data3_all = {'y_data':y_data3,'u_data':u_data3}

# #Save sammple imdices to file
# N = 25
# f = open('/app/current_output/example3_indices_n='+str(N)+'.csv', 'w')
# N_tot = np.size(u_data3)
# fw = csv.writer(f)
# header =[*["optim"]*(int(N*4/5)),*["valid"]*(int(N/5))]
# np.size(header)
# fw.writerow(header)
# for i in range(100):
#     fw.writerow(random.sample(range(N_tot),N))
# f.close()

#Set parameters
d = 52
p = 2

importlib.reload(ro)
#Run generative model/sparse optimizations
n1 = 25
indices1 = pd.read_csv('/app/paper_optimizations_final/example3/example3_indices_n='+str(n1)+'.csv')
fn1 = 'data3_n=' + str(n1)
ro.run_l1(0,99,fn1,d,p,data3_all,indices1,lasso_eps=1e-10,lasso_iter=1e5,lasso_tol=1e-4,lasso_n_alphas=100)
ro.run_genmod(69,99,fn1,d,p,data3_all,indices1,lasso_eps=1e-10,lasso_iter=1e5,lasso_tol=1e-4,lasso_n_alphas=100)

# %%
import pickle
import csv
mi_mat = pcu.make_mi_mat(d,p)
Psi = pcu.make_Psi(y_data3,mi_mat)
# %%
fn = '/app/current_output/data3_n=25'
n = 25
indices1 = pd.read_csv('/app/paper_optimizations_final/example3/example3_indices_n=25.csv')
nsamp = 50
dmo_err = np.zeros(nsamp)
las_err = np.zeros(nsamp)
for i in range(nsamp):
    #print(i)
    dmo = pickle.load(open(fn + '_' + str(i) + '.pkl','rb'))
    data_indices = indices1.loc[i].to_numpy()
    not_data_indices = np.setdiff1d(range(np.size(u_data3)),data_indices)
    #print(dmo.z)
    # opening the CSV file
    # las = pickle.load(open('/app/paper_optimizations/example3/data3_n=135_wlas3_'+str(i) + '.pkl','rb'))
    # las_coef = las.coef_
    las = pd.read_csv('/app/current_output/data3_n=25_omp_' + str(i)+'.csv')
    las_coef = las.iloc[:,0].to_numpy()
    #print(las_coef)
    #print(np.linalg.norm(Psi[not_data_indices,:]@dmo.c-u_data3[not_data_indices])/np.linalg.norm(u_data3[not_data_indices]))
    dmo_err[i] = np.linalg.norm(Psi[not_data_indices]@dmo.c-u_data3[not_data_indices])/np.linalg.norm(u_data3[not_data_indices])
    las_err[i] = np.linalg.norm(Psi[not_data_indices]@las_coef-u_data3[not_data_indices])/np.linalg.norm(u_data3[not_data_indices])
    #print(las_err)
    #print(dmo_err)
    #print((las_err-dmo_err)/las_err)
    #print(Psi[not_data_indices,:]@las_coef-u_data3[not_data_indices]-(Psi[not_data_indices]@dmo.c-u_data3[not_data_indices]))

import plotly.graph_objects as go
fig = go.Figure(go.Histogram(
    x=las_err,
    xbins=dict(start=0,end=1,size=.04)))
fig.add_trace(go.Histogram(
    x=dmo_err,
    xbins=dict(start=0,end=1,size=.04)))
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.5)
fig.show()

fig = go.Figure(go.Histogram(x=(las_err-dmo_err)/las_err))
fig.show()

fig = go.Figure(go.Histogram(x=(dmo_err-las_err)/dmo_err))
fig.show()

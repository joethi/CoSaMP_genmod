
# load package
import scipy.io as sio
import numpy as np
#import sys
import random
import pandas
#import pickle
from scipy.special import factorial
import importlib
import colorsys

#Import local functions
#import genmodpce.figures.paper_figures as pf
#import genmodpce.presentation_figures as prf

#from genmodpce import decay_models as dm

#NEED TO LOAD DECAY MODELS LIKE THIS FOR IT TO UNPICKLE
import sys
sys.path.append('/app/genmodpce/')
sys.path.append('/app/genmodpce/figures')

import paper_figures as pf
import decay_models
import pickle
import pandas as pd

importlib.reload(pf)
importlib.reload(decay_models)

# %% Dataset 1
y_data1 = np.genfromtxt('/app/data/dataset1/xi_40000_1.dat')
u_data1 = np.genfromtxt('/app/data/dataset1/fsamp_40000_1.dat')
data1_all = {'y_data':y_data1,'u_data':u_data1}

#Set parameters
N = 40
p = 3
d = 14
P =int(factorial(d+p)/(factorial(d)*factorial(p)))

#Plot coefficient and reconstruction error
fn = '/app/paper_optimizations_final/example1/n='+str(N)+'/'
fn_c_ls = '/app/paper_optimizations_final/example1/data1_ls_coefs_N=40000.csv'
c_ls = pd.read_csv(fn_c_ls).to_numpy().flatten()
indices1 = pd.read_csv(fn+'/example1_indices_n='+str(N)+'.csv')
importlib.reload(pf)
pf.plot_coef_and_recons(50,fn,'data1_n='+str(N),u_data1,y_data1,indices1,1000,bin_num=15,plotTrain=False,plotCoefs=True,c_ls=c_ls)

#Plot error as a function of N
fn = '/app/paper_optimizations_final/example1/'
pf.plot_error_with_N([30,40,80,160,320],fn,50,1000,u_data1,y_data1,'data1',indices1,c_ls=c_ls,plot_coefs=True,plot_recons=True)



# %% Dataset 2
y_dat = sio.loadmat('/app/data/dataset2/y_samples.mat')
u_dat = sio.loadmat('/app/data/dataset2/u_samples.mat')
y_data2 = y_dat['y_samples']
u_data2 = u_dat['u_samples'].flatten()
data2_all = {'y_data':y_data2,'u_data':u_data2}

np.size(u_data1)

#Set parameters
d = 20
p = 3
N = 60

fn = '/app/paper_optimizations_final/example2/'
fn_c_ls = '/app/paper_optimizations/example2/data2_ls_coefs'
fn_c_ls = '/app/paper_optimizations_final/example2/data2_ls_coefs_N=30000.csv'
c_ls_df = pd.read_csv(fn_c_ls)
c_ls = c_ls_df.iloc[:,1].to_numpy().flatten()
indices2 = pd.read_csv(fn+'/example2_indices_n='+str(N)+'.csv')
pf.plot_coef_and_recons(50,fn,'data2_n='+str(N),u_data2,y_data2,indices2,1000,bin_num=15,plotCoefs=True,c_ls=c_ls)

total_correct_flips = 0
total_incorrect_flips = 0
for i in range(50):
    print(i)
    dmo = pickle.load(open(fn + '/GenMod/data2_n=' + str(N) + '_' + str(i) + '.pkl','rb'))

    las = pd.read_csv(fn + '/IRW-Lasso/data2_n=' + str(N) + '_las_' + str(i)+'.csv')
    c_las = las.iloc[:,0].to_numpy()

    omp = pd.read_csv(fn + '/OMP/data2_n=' + str(N) + '_omp_' + str(i)+'.csv')
    c_omp = omp.iloc[:,0].to_numpy()

    [flip_correct, flip_incorrect, max_incorrect] = pf.plot_coef_comparison(dmo.c,c_omp,c_las,dmo.zeta,c_ls,'2')
    total_correct_flips += flip_correct
    total_incorrect_flips += flip_incorrect

(total_correct_flips+total_incorrect_flips)/50
total_correct_flips/(total_incorrect_flips+total_correct_flips)

i=15
dmo = pickle.load(open(fn + '/GenMod/data2_n=' + str(N) + '_' + str(i) + '.pkl','rb'))
las = pd.read_csv(fn + '/IRW-Lasso/data2_n=' + str(N) + '_las_' + str(i)+'.csv')
c_las = las.iloc[:,0].to_numpy()
omp = pd.read_csv(fn + '/OMP/data2_n=' + str(N) + '_omp_' + str(i)+'.csv')
c_omp = omp.iloc[:,0].to_numpy()
[flip_correct, flip_incorrect, max_incorrect] = pf.plot_coef_comparison(dmo.c,c_omp,c_las,dmo.zeta,c_ls,'2')

# %% Dataset 3
y_dat = sio.loadmat('/app/data/dataset3/xi256.mat')
u_dat = sio.loadmat('/app/data/dataset3/U128.mat')
y_data3_temp = y_dat['xi']
u_data3_tot = u_dat['U']

y_data3 =np.copy(y_data3_temp)
y_data3[:,1]=2*(y_data3_temp[:,1]-105)/(np.max(y_data3_temp[:,1])-105)-1
y_data3[:,0]=2*(y_data3_temp[:,0]-.004)/(.01-.004)-1
np.max(y_data3[:,1])
np.min(y_data3[:,1])


Ntot=np.size(u_data3_tot,1)
#Get output data from center of domain (grid w/out boundary is 126, so pick 63)
u_data3 = u_data3_tot[62,:]

np.size(u_data3)
data3_all = {'y_data':y_data3,'u_data':u_data3}

#Set parameters
d = 52
p = 2

importlib.reload(pf)
N=25
fn = '/app/paper_optimizations_final/example3/n='+str(object=N)+'/'
indices3 = pd.read_csv(fn+'/example3_indices_n='+str(N)+'.csv')
pf.plot_coef_and_recons(100,fn,'data3_n='+str(N),u_data3,y_data3,indices3,4,bin_num=8,plotNoSparse=False,plotCoefs=False)


N=50
fn = '/app/paper_optimizations_final/example3/n='+str(object=N)+'/'
indices3 = pd.read_csv(fn+'/example3_indices_n='+str(N)+'.csv')
pf.plot_coef_and_recons(100,fn,'data3_n='+str(N),u_data3,y_data3,indices3,Ntot-N,bin_num=12,plotNoSparse=False,plotCoefs=False)



#pf.plot_mean_error(50,fn,fn_c_ls,'example1',u_data1,y_data1,60,bin_num=15)

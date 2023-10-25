
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

import matplotlib as mpl
import matplotlib.pyplot as plt
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
import decay_model_utils as dmu
importlib.reload(pf)
importlib.reload(decay_models)


# %% Dataset 1
#Set parameters
N = 60
d = 20#14
fn = '/app/paper_optimizations_final/example2/'
dataset='data2_n='+str(N)
i = 3 #6
dmo = pickle.load(open(fn + '/GenMod/' + dataset + '_' + str(i) + '.pkl','rb'))

np.size(dmo.z)
plt.figure()
plt.scatter(x=range(d),y=dmo.z[1:1+d])
plt.scatter(x=range(d),y=dmo.z[d+1:])
plt.legend(['Exponential decay rate'],['Algebraic decay rate'])
plt.show()

# %%

np.min(np.absolute(dmo.z[1:]))


loss_hist=[]
loss_hist_v=[]
total_iter=0
iter_break=[]
for j in range(np.size(dmo.nuhist,0)):
    zhist = dmo.zdict['Iter:'+str(j)]
    nu = dmo.nuhist[j]
    zeta = dmo.zeta_hist[j]
    for z in zhist:
        G = dmu.decay_model(z,dmo.mi_c,dmo.beta_param_mat)
        c = nu + G * zeta
        loss = np.linalg.norm(dmo.Psi @ c - dmo.u_data)/np.linalg.norm(dmo.u_data)
        loss_v = np.linalg.norm(dmo.Psi_valid @ c - dmo.u_valid)/np.linalg.norm(dmo.u_valid)
        loss_hist.append(loss)
        loss_hist_v.append(loss_v)

        total_iter+=1
    iter_break.append(total_iter)

# %%

plt.figure(figsize=(4,3),dpi=600)
plt.plot(range(np.size(loss_hist)),loss_hist)
plt.plot(range(np.size(loss_hist)),loss_hist_v)
for ii in range(len(iter_break)):
    plt.axvline(x=iter_break[ii],linestyle='dashed',color='black')
    if ii==0:
        plt.axhline(y=min(loss_hist_v),xmax=np.where(loss_hist_v==min(loss_hist_v))[0][0]/total_iter,color='red',linestyle='dotted')
        plt.axvline(
            x=np.where(loss_hist_v==min(loss_hist_v))[0][0],
            ymax=.16,color='red',linestyle='dotted')
plt.legend(['Optimization data','Validation data','IRW-Lasso step', 'Minimum validation error'],bbox_to_anchor=(.03, 1), loc='upper left')


plt.semilogy()
plt.ylabel('Relative L2 Loss')
plt.xlabel('Iteration')
plt.xlim([0,total_iter])
plt.ylim([0.01,4])

plt.savefig('/app/current_output/fig_loss_history_example_2.pdf')
plt.show()

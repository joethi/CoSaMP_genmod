import numpy as np
import numpy.linalg as la
from sklearn.model_selection import KFold
import random
import matplotlib.pyplot as plt
def cosamp_func(Psi,u_data,S_val,max_iter=10,hlt_crit='iter',tol_res=1e-4):
    res = np.copy(u_data)
    P_vl = np.size(Psi,1)
    k = 0; halt = False
    res_norm = []
    while not halt:
        c_2s = np.zeros(P_vl)
        c_s = np.zeros(P_vl)
        c_prx = Psi.T @ res
        Omg = (np.argsort(np.abs(c_prx))[::-1])[:2*S_val]
        if k==0:
            T_set = Omg 
        else:
            T_set = np.union1d(Omg,Lam)
        #import pdb; pdb.set_trace()    
        c_2s_tmp = (la.inv(Psi[:,T_set].T @ Psi[:,T_set]) @ Psi[:,T_set].T @ u_data).flatten()
        c_2s[T_set] = c_2s_tmp 
        Lam = (np.argsort(np.abs(c_2s))[::-1])[:S_val]
        c_s[Lam] = c_2s[Lam]
        res = u_data - Psi[:,Lam] @ c_s[Lam] 
        res_norm.append(la.norm(res))
        if hlt_crit=='norm':
            halt = (res_norm[k] <= tol_res)
        elif hlt_crit=='iter':
            halt = (k==max_iter-1)
        print('k:\n',k)
        print('halting criterion:\n',halt)
        print('residual norm:', res_norm[k])
        k += 1
    return c_s, res_norm

def cross_valid_cosamp(Psi,u_data,S_rng,csmp_prms,n_fold=5):
     max_it_cs = csmp_prms['maxit_csmp'] 
     hltcrit_cs = csmp_prms['hlcrt_csmp']
     tolres_cs = csmp_prms['tolres_csmp']
     rnd_st_cvcs = csmp_prms['rnd_st_cvcs'] 
     kf = KFold(n_splits=n_fold,shuffle=True,random_state=rnd_st_cvcs)
     res_trn_fld = np.zeros((len(S_rng),n_fold))
     res_tst_fld = np.zeros_like(res_trn_fld)
     for j,S_val in enumerate(S_rng):
         for i, (trn_ind_fld,tst_ind_fld) in enumerate(kf.split(Psi)): 
             print(f'---Fold:{i}------') 
             c_s,res_fl = cosamp_func(Psi[trn_ind_fld,:],u_data[trn_ind_fld],S_val,max_iter=10,hlt_crit='iter',tol_res=1e-4)           
             #import pdb; pdb.set_trace()
             res_trn_fld[j,i] = res_fl[-1]
             res_tst_fld[j,i] = la.norm(u_data[tst_ind_fld] - Psi[tst_ind_fld,:] @ c_s)       
      #Averaging the validation errors over the folds for each S and picking the S that minimizes val error::
     mn_vlerr = np.mean(res_tst_fld,axis=1)  
     min_ind = np.argmin(mn_vlerr)
     S_opt = S_rng[min_ind]
     #optional step: using the optimal sparsity do the full training.
     c_s_opt,res_fl_opt = cosamp_func(Psi,u_data,S_val,max_iter=10,hlt_crit='iter',tol_res=1e-4)           
     import pdb; pdb.set_trace()

     return S_opt,c_s_opt
#====================================================================
#================Testing the cross-validation algorithm==============
#====================================================================
out_dir = "../output/titan_ppr/Scosamp_tst"
P = 2**13; N = 2**10
Psi = np.random.randn(N,P) 
c_z = np.zeros(P)
rnd_sel_n1 = random.sample(list(range(P)),50)
import pdb; pdb.set_trace()
rnd_sel_p1_smp = np.setdiff1d(list(range(P)),rnd_sel_n1)
rnd_sel_p1 = random.sample(rnd_sel_p1_smp.tolist(),70)
c_z[rnd_sel_n1] = -1.0
c_z[rnd_sel_p1] = 1.0
S_val = np.size(np.nonzero(c_z)[0])
print('rnd_sel_n1',rnd_sel_n1)
print('rnd_sel_p1',rnd_sel_p1)
u_data = Psi @ c_z
plt.figure(1)
plt.plot(list(range(P)),c_z,label='c_z')
plt.savefig(f'{out_dir}/signal_N{N}_P{P}.png')
import pdb; pdb.set_trace()
csmp_prms = {'maxit_csmp':10,'hlcrt_csmp':'iter','tolres_csmp':1e-4,'rnd_st_cvcs': 1}
S_rng = list(range(100,140,5))
S_opt,c_opt = cross_valid_cosamp(Psi,u_data,S_rng,csmp_prms,n_fold=5)
import pdb; pdb.set_trace()
#plt.plot(list(range()),c_z,label='')
#c_csmp, res = cosamp_func(Psi,u_data,S_val,max_iter=10,hlt_crit='iter',tol_res=1e-4)
c_csmp_plt = np.zeros(P)
plt.figure(2)
plt.plot(list(range(P)),c_z,label='original')
plt.plot(list(range(P)),c_opt,'--',label='CoSaMP')
plt.legend()
plt.ylabel('signal')
plt.savefig(f'{out_dir}/cmp_signal_N{N}_P{P}.png')
plt.figure(3)
plt.plot(list(range(N)),u_data,label='original')
plt.plot(list(range(N)),Psi @ c_opt,'--',label='CoSaMP')
plt.legend()
plt.savefig(f'{out_dir}/cmp_qoi_N{N}_P{P}.png')
plt.ylabel('signal')
#====================================================================
#================Testing the algorithm============================
#====================================================================
#============zero-one signal======================================
#out_dir = "../output/titan_ppr/Scosamp_tst"
#P = 2**13; N = 2**10
#Psi = np.random.randn(N,P) 
#c_z = np.zeros(P)
#rnd_sel_n1 = random.sample(list(range(P)),50)
#import pdb; pdb.set_trace()
#rnd_sel_p1_smp = np.setdiff1d(list(range(P)),rnd_sel_n1)
#rnd_sel_p1 = random.sample(rnd_sel_p1_smp.tolist(),70)
#c_z[rnd_sel_n1] = -1.0
#c_z[rnd_sel_p1] = 1.0
#S_val = np.size(np.nonzero(c_z)[0])
#print('rnd_sel_n1',rnd_sel_n1)
#print('rnd_sel_p1',rnd_sel_p1)
#u_data = Psi @ c_z
#plt.figure(1)
#plt.plot(list(range(P)),c_z,label='c_z')
#plt.savefig(f'{out_dir}/signal_N{N}_P{P}.png')
#import pdb; pdb.set_trace()
##plt.plot(list(range()),c_z,label='')
#c_csmp, res = cosamp_func(Psi,u_data,S_val,max_iter=10,hlt_crit='iter',tol_res=1e-4)
#c_csmp_plt = np.zeros(P)
#plt.figure(2)
#plt.plot(list(range(P)),c_z,label='original')
#plt.plot(list(range(P)),c_csmp,'--',label='CoSaMP')
#plt.legend()
#plt.ylabel('signal')
#plt.savefig(f'{out_dir}/cmp_signal_N{N}_P{P}.png')
#plt.figure(3)
#plt.plot(list(range(N)),u_data,label='original')
#plt.plot(list(range(N)),Psi @ c_csmp,'--',label='CoSaMP')
#plt.legend()
#plt.savefig(f'{out_dir}/cmp_qoi_N{N}_P{P}.png')
#plt.ylabel('signal')

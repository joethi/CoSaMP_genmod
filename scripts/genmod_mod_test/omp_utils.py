import sklearn.linear_model as lm
from  sklearn.model_selection import KFold
import genmod_mod_test.polynomial_chaos_utils as pcu
import numpy as np
import numpy.linalg as la
import genmod_mod_test.train_NN_omp_wptmg as tnn
import genmod_mod_test.test_coeffs_val_er_utils as tcu
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing
from functools import partial
import os
import sys
    # Find initial signs with orthogonal matching pursuit (sklearn):
def omp_utils_order_ph(out_dir_ini,d,p,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_omp,j):
    d_omp = d
    p_omp = p
    N = len(optim_indices)
    #OMPCV, and without CV:
    mi_mat_omp = pcu.make_mi_mat(d_omp, p_omp)
    P_omp = np.size(mi_mat_omp,0)
    Psi_time_strt = time.time()
    Psi_omp = pcu.make_Psi(y_data[optim_indices,:d],mi_mat_omp,chc_Psi) 
    Psi_time_end = time.time()
    print('Psi_time for ph', Psi_time_end - Psi_time_strt)
    #import pdb;pdb.set_trace()
    if chc_omp_slv=='ompcv':
        omp = lm.OrthogonalMatchingPursuitCV(cv=5, fit_intercept=False)   
    elif chc_omp_slv=='stdomp':    
    # S_omp0 = 26
        omp = lm.OrthogonalMatchingPursuit(n_nonzero_coefs=S_omp,fit_intercept=False)
    #    import pdb; pdb.set_trace() 
    print('omp:',omp)
    omp.fit(Psi_omp, u_data[optim_indices].flatten())
    #  warnings.filterwarnings('error')
    #  try:
    #      omp.fit(Psi_omp, u_data[optim_indices].flatten())
    #  except Warning as w:
    #      print("Warning:", str(w))
    #      import pdb;pdb.set_trace()
    c_om_std = omp.coef_ 
    c_ini = c_om_std
    #    c_ini[74] = 0.001
    #    c_ini[9] = 0
    S_omp = np.size(np.nonzero(c_ini)[0])
    print('S_0:', S_omp)
    # c_ini = pd.read_csv(f'{out_dir_ini}/ini/ompcs/cini_1dellps_n=680_genmod_S=168_0.csv').to_numpy().flatten()
    # Test on the data:
    train_err_p, valid_err_p = tnn.val_test_err(data_tst,mi_mat_omp,c_ini)
    # Testing error: is the error on the training data:
    print(f'Training Error for c_ini: {train_err_p}')
    # Validation error: is the error on the unseen data:
    print(f'Validation Error for c_ini: {valid_err_p}')
    df_comp0 = pd.DataFrame({'c_omp':c_ini})
    df_comp0.to_csv(f'{out_dir_ini}/plots/j={j}/comp_1dellps_n={N}_genmod_S={S_omp}_p={p_omp}_j{j}.csv',index=False)
    df_omp_err = pd.DataFrame({'valid_err':[valid_err_p],'test_err':[train_err_p]})
    df_omp_err.to_csv(f'{out_dir_ini}/plots/j={j}/epsu_1dellps_n={N}_genmod_S={S_omp}_p={p_omp}.csv',index=False)
    ##==========================================================================================================
    return c_ini, S_omp, train_err_p, valid_err_p, P_omp, mi_mat_omp, Psi_omp
def omp_utils_order_ph_prllNcalc(out_dir_ini,d,p,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_omp,j):
    d_omp = d
    p_omp = p
    N = len(optim_indices)
    #OMPCV, and without CV:
    mi_mat_omp = pcu.make_mi_mat(d_omp, p_omp)
    P_omp = np.size(mi_mat_omp,0)
    Psi_omp = np.zeros((N,P_omp))
    Psi_time_strt = time.time()
    #Psi_omp = pcu.make_Psi_prll(y_data[optim_indices,:d],mi_mat_omp,chc_Psi) 
    #calculate Psi in a parallel manner: 
    num_work_psi = 15
    Psi_omp_part = partial(pcu.make_Psi_prll,y_data[optim_indices,:d],mi_mat_omp,chc_Psi) 
    pool = multiprocessing.Pool(processes=num_work_psi)
    results_Psi = pool.map(Psi_omp_part,list(range(N)))
    #Psi_omp_srl = pcu.make_Psi(y_data[optim_indices,:d],mi_mat_omp,chc_Psi)
    pool.close()
    pool.join()
    #import pdb;pdb.set_trace()
    for i in range(N):
        Psi_omp[i,:] = results_Psi[i] 
    Psi_time_end = time.time()
    print('Psi_time for ph', Psi_time_end - Psi_time_strt)
    #import pdb;pdb.set_trace()
    if chc_omp_slv=='ompcv':
        omp = lm.OrthogonalMatchingPursuitCV(cv=5, fit_intercept=False)   
    elif chc_omp_slv=='stdomp':    
    # S_omp0 = 26
        omp = lm.OrthogonalMatchingPursuit(n_nonzero_coefs=S_omp,fit_intercept=False)
    #    import pdb; pdb.set_trace() 
    print('omp:',omp)
    omp.fit(Psi_omp, u_data[optim_indices].flatten())
    #  warnings.filterwarnings('error')
    #  try:
    #      omp.fit(Psi_omp, u_data[optim_indices].flatten())
    #  except Warning as w:
    #      print("Warning:", str(w))
    #      import pdb;pdb.set_trace()
    c_om_std = omp.coef_ 
    c_ini = c_om_std
    #    c_ini[74] = 0.001
    #    c_ini[9] = 0
    S_omp = np.size(np.nonzero(c_ini)[0])
    print('S_0:', S_omp)
    # c_ini = pd.read_csv(f'{out_dir_ini}/ini/ompcs/cini_1dellps_n=680_genmod_S=168_0.csv').to_numpy().flatten()
    # Test on the data:
    train_err_p, valid_err_p = tnn.val_test_err(data_tst,mi_mat_omp,c_ini)
    # Testing error: is the error on the training data:
    print(f'Training Error for c_ini: {train_err_p}')
    # Validation error: is the error on the unseen data:
    print(f'Validation Error for c_ini: {valid_err_p}')
    df_comp0 = pd.DataFrame({'c_omp':c_ini})
    df_comp0.to_csv(f'{out_dir_ini}/plots/j={j}/comp_1dellps_n={N}_genmod_S={S_omp}_p={p_omp}_j{j}.csv',index=False)
    df_omp_err = pd.DataFrame({'valid_err':[valid_err_p],'test_err':[train_err_p]})
    df_omp_err.to_csv(f'{out_dir_ini}/plots/j={j}/epsu_1dellps_n={N}_genmod_S={S_omp}_p={p_omp}.csv',index=False)
    ##==========================================================================================================
    return c_ini, S_omp, train_err_p, valid_err_p, P_omp, mi_mat_omp, Psi_omp
def omp_utils_order_ph_prll(out_dir_ini,indices0,d,p,y_data,u_data,chc_Psi,chc_omp_slv,S_omp,Nv,j):
    optim_indices = indices0.iloc[j].to_numpy()
    valid_indices = np.setdiff1d(range(np.size(u_data)), optim_indices)
    trains = [name for name in indices0.columns if name.startswith("optim")]
    test_indices = indices0.loc[j][trains].to_numpy()
    data_tst = {'y_data':y_data,'u_data':u_data,'val_ind':valid_indices,'test_ind':test_indices,'opt_ind':optim_indices,'Nv':Nv,
            'chc_poly':chc_Psi,'chc_omp':chc_omp_slv}
    d_omp = d
    p_omp = p
    N = len(optim_indices)
    #OMPCV, and without CV:
    mi_mat_omp = pcu.make_mi_mat(d_omp, p_omp)
    P_omp = np.size(mi_mat_omp,0)
    Psi_omp = pcu.make_Psi(y_data[optim_indices,:d],mi_mat_omp,chc_Psi) 
    #import pdb;pdb.set_trace()
    if chc_omp_slv=='ompcv':
        omp = lm.OrthogonalMatchingPursuitCV(cv=5, fit_intercept=False)   
    elif chc_omp_slv=='stdomp':    
    # S_omp0 = 26
        omp = lm.OrthogonalMatchingPursuit(n_nonzero_coefs=S_omp,fit_intercept=False)
    #    import pdb; pdb.set_trace() 
    print('omp:',omp)
    omp.fit(Psi_omp, u_data[optim_indices].flatten())
    #  warnings.filterwarnings('error')
    #  try:
    #      omp.fit(Psi_omp, u_data[optim_indices].flatten())
    #  except Warning as w:
    #      print("Warning:", str(w))
    #      import pdb;pdb.set_trace()
    c_om_std = omp.coef_ 
    c_ini = c_om_std
    #    c_ini[74] = 0.001
    #    c_ini[9] = 0
    S_omp = np.size(np.nonzero(c_ini)[0])
    print('S_0:', S_omp)
    # c_ini = pd.read_csv(f'{out_dir_ini}/ini/ompcs/cini_1dellps_n=680_genmod_S=168_0.csv').to_numpy().flatten()
    # Test on the data:
    train_err_p, valid_err_p = tnn.val_test_err(data_tst,mi_mat_omp,c_ini)
    # Testing error: is the error on the training data:
    print(f'Training Error for c_ini: {train_err_p}')
    # Validation error: is the error on the unseen data:
    print(f'Validation Error for c_ini: {valid_err_p}')
    df_comp0 = pd.DataFrame({'c_omp':c_ini})
    df_comp0.to_csv(f'{out_dir_ini}/plots/j={j}/comp_1dellps_n={N}_genmod_S={S_omp}_p={p_omp}_j{j}.csv',index=False)
    df_omp_err = pd.DataFrame({'valid_err':[valid_err_p],'test_err':[train_err_p]})
    df_omp_err.to_csv(f'{out_dir_ini}/plots/j={j}/epsu_1dellps_n={N}_genmod_S={S_omp}_p={p_omp}.csv',index=False)
    ##==========================================================================================================
    return {"c":c_ini, "S":S_omp, "trn_err":train_err_p, "val_err":valid_err_p, "P":P_omp, "mi_mat":mi_mat_omp, 'Psi_omp':Psi_omp}

def omp_err_for_var_S_prllNclc(S_true_78,S_strt,out_dir_ini,indices0,d,p_0,y_data,u_data,chc_Psi,chc_omp_slv,Nv,chc_lo,mi_mat,N,j):
    os.makedirs(f'{out_dir_ini}/plots/j={j}',exist_ok=True)    
    optim_indices = indices0.iloc[j].to_numpy()
    valid_indices = np.setdiff1d(range(np.size(u_data)), optim_indices)
    trains = [name for name in indices0.columns if name.startswith("optim")]
    test_indices = indices0.loc[j][trains].to_numpy()
    data_tst = {'y_data':y_data,'u_data':u_data,'val_ind':valid_indices,'test_ind':test_indices,'opt_ind':optim_indices,'Nv':Nv,
            'chc_poly':chc_Psi,'chc_omp':chc_omp_slv}
    val_omp_tot = []
    trn_omp_tot = []    
    for ls_ind in range(len(S_true_78)-S_strt+1):
        if chc_lo=='SO':
            #c_ini, S_omp0, train_err_p0, valid_err_p0,P_omp,mi_mat_omp, Psi_omp = omp_utils_order_ph(out_dir_ini,d,p_0,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_strt+ls_ind,j)
            c_ini, S_omp0, train_err_p0, valid_err_p0,P_omp,mi_mat_omp, Psi_omp = omp_utils_order_ph_prllNcalc(out_dir_ini,d,p_0,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_strt+ls_ind,j)
            val_omp_tot.append(valid_err_p0)
            trn_omp_tot.append(train_err_p0)
        elif chc_lo=='LS':
            P = int(np.math.factorial(d+p_0)/(np.math.factorial(d)*np.math.factorial(p_0)))
            c_ls = tcu.apply_lst_sqr_actset(S_true_78[:S_strt+ls_ind],P,mi_mat,chc_Psi,u_data,y_data,optim_indices)
            trn_err_ls, valid_err_ls = tnn.val_test_err(data_tst,mi_mat,c_ls)
            S_omp0 = np.size(np.nonzero(c_ls)[0])
            df_comp0 = pd.DataFrame({'coeff':c_ls})
            df_comp0.to_csv(f'{out_dir_ini}/plots/j={j}/cls_1dellps_n={N}_genmod_S={S_omp0}_p={p_0}_j{j}.csv',index=False)
            val_omp_tot.append(valid_err_ls)
            trn_omp_tot.append(trn_err_ls)
    df_err_ls = pd.DataFrame({'val':np.array(val_omp_tot),'trn':np.array(trn_omp_tot)})
    df_err_ls.to_csv(f'{out_dir_ini}/plots/trnval_err_list_j{j}.csv',index=False)
    plt.figure(j+1)
    plt.plot(np.arange(S_strt,len(S_true_78)+1,1),np.array(val_omp_tot),'--b*',label='valid,j={j}')
    plt.plot(np.arange(S_strt,len(S_true_78)+1,1),np.array(trn_omp_tot),'--r*',label='train,j={j}')
    plt.xlabel('Sparsity')
    plt.ylabel('Relative valdiation error')
    plt.savefig(f'{out_dir_ini}/plots/trnvalerr_vs_S_j={j}.png',dpi=300)        

    return {"val_err":val_omp_tot,"trn_err":trn_omp_tot,'j':j,'S':S_omp0} 
def omp_err_for_var_S(S_true_78,S_strt,out_dir_ini,indices0,d,p_0,y_data,u_data,chc_Psi,chc_omp_slv,Nv,chc_lo,mi_mat,N,j):
    os.makedirs(f'{out_dir_ini}/plots/j={j}',exist_ok=True)    
    optim_indices = indices0.iloc[j].to_numpy()
    valid_indices = np.setdiff1d(range(np.size(u_data)), optim_indices)
    trains = [name for name in indices0.columns if name.startswith("optim")]
    test_indices = indices0.loc[j][trains].to_numpy()
    data_tst = {'y_data':y_data,'u_data':u_data,'val_ind':valid_indices,'test_ind':test_indices,'opt_ind':optim_indices,'Nv':Nv,
            'chc_poly':chc_Psi,'chc_omp':chc_omp_slv}
    val_omp_tot = []
    trn_omp_tot = []    
    for ls_ind in range(len(S_true_78)-S_strt+1):
        if chc_lo=='SO':
            c_ini, S_omp0, train_err_p0, valid_err_p0,P_omp,mi_mat_omp, Psi_omp = omp_utils_order_ph(out_dir_ini,d,p_0,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_strt+ls_ind,j)
            val_omp_tot.append(valid_err_p0)
            trn_omp_tot.append(train_err_p0)
        elif chc_lo=='LS':
            P = int(np.math.factorial(d+p_0)/(np.math.factorial(d)*np.math.factorial(p_0)))
            c_ls = tcu.apply_lst_sqr_actset(S_true_78[:S_strt+ls_ind],P,mi_mat,chc_Psi,u_data,y_data,optim_indices)
            trn_err_ls, valid_err_ls = tnn.val_test_err(data_tst,mi_mat,c_ls)
            S_omp0 = np.size(np.nonzero(c_ls)[0])
            df_comp0 = pd.DataFrame({'coeff':c_ls})
            df_comp0.to_csv(f'{out_dir_ini}/plots/j={j}/cls_1dellps_n={N}_genmod_S={S_omp0}_p={p_0}_j{j}.csv',index=False)
            val_omp_tot.append(valid_err_ls)
            trn_omp_tot.append(trn_err_ls)
    df_err_ls = pd.DataFrame({'val':np.array(val_omp_tot),'trn':np.array(trn_omp_tot)})
    df_err_ls.to_csv(f'{out_dir_ini}/plots/trnval_err_list_j{j}.csv',index=False)
    plt.figure(j+1)
    plt.plot(np.arange(S_strt,len(S_true_78)+1,1),np.array(val_omp_tot),'--b*',label='valid,j={j}')
    plt.plot(np.arange(S_strt,len(S_true_78)+1,1),np.array(trn_omp_tot),'--r*',label='train,j={j}')
    plt.xlabel('Sparsity')
    plt.ylabel('Relative valdiation error')
    plt.savefig(f'{out_dir_ini}/plots/trnvalerr_vs_S_j={j}.png',dpi=300)        

    return {"val_err":val_omp_tot,"trn_err":trn_omp_tot,'j':j,'S':S_omp0} 

def omp_utils_order_ph_dummy(out_dir_ini,cht_fl_rpdc,d,p,y_data,u_data,data_tst,optim_indices,chc_Psi,chc_omp_slv,S_omp,j):
    d_omp = d
    p_omp = p
    N = len(optim_indices)
    #OMPCV, and without CV:
    mi_mat_omp = pcu.make_mi_mat(d_omp, p_omp)
    P_omp = np.size(mi_mat_omp,0)
    Psi_omp = pcu.make_Psi(y_data[optim_indices,:d],mi_mat_omp,chc_Psi) 
    c_ini = pd.read_csv(f"{cht_fl_rpdc}").to_numpy().flatten()
    #    c_ini[74] = 0.001
    #    c_ini[9] = 0
    S_omp = np.size(np.nonzero(c_ini)[0])
    print('p',p,'S:', S_omp)
    # c_ini = pd.read_csv(f'{out_dir_ini}/ini/ompcs/cini_1dellps_n=680_genmod_S=168_0.csv').to_numpy().flatten()
    # Test on the data:
    train_err_p, valid_err_p = tnn.val_test_err(data_tst,mi_mat_omp,c_ini)
    # Testing error: is the error on the training data:
    print(f'Training Error for c_ini: {train_err_p}')
    # Validation error: is the error on the unseen data:
    print(f'Validation Error for c_ini: {valid_err_p}')
    df_comp0 = pd.DataFrame({'c_omp':c_ini})
    df_comp0.to_csv(f'{out_dir_ini}/plots/j={j}/comp_1dellps_n={N}_genmod_S={S_omp}_p={p_omp}_j{j}.csv',index=False)
    df_omp_err = pd.DataFrame({'valid_err':[valid_err_p],'test_err':[train_err_p]})
    df_omp_err.to_csv(f'{out_dir_ini}/plots/j={j}/epsu_1dellps_n={N}_genmod_S={S_omp}_p={p_omp}.csv',index=False)
    ##==========================================================================================================
    return c_ini, S_omp, train_err_p, valid_err_p, P_omp, mi_mat_omp, Psi_omp
#Cosamp utils
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
        print('shape of T_set:',T_set.shape)
        k += 1
        #import pdb; pdb.set_trace()
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
             c_s,res_fl = cosamp_func(Psi[trn_ind_fld,:],u_data[trn_ind_fld],S_val,max_iter=max_it_cs,hlt_crit=hltcrit_cs,tol_res=tolres_cs)           
             #import pdb; pdb.set_trace()
             res_trn_fld[j,i] = res_fl[-1]
             res_tst_fld[j,i] = la.norm(u_data[tst_ind_fld] - Psi[tst_ind_fld,:] @ c_s)       
      #Averaging the validation errors over the folds for each S and picking the S that minimizes val error::
     mn_vlerr = np.mean(res_tst_fld,axis=1)  
     mn_trnerr = np.mean(res_trn_fld,axis=1)
     min_ind = np.argmin(mn_vlerr)
     S_opt = S_rng[min_ind]
     #optional step: using the optimal sparsity do the full training.
     c_s_opt,res_fl_opt= cosamp_func(Psi,u_data,S_val,max_iter=10,hlt_crit='iter',tol_res=1e-4)           
     #import pdb; pdb.set_trace()

     return S_opt,c_s_opt,mn_vlerr,mn_trnerr
     
#   
##Calculate S for 0 and h:
#    if chc_eps == 'c':
#        S_refh = np.size(np.nonzero(c_ref[:P])[0])
#        S_ref0 = np.size(np.nonzero(c_ref[:P_omp])[0])
#    # c_ini = c_gen[:P_omp]
#=============================================================================
# #p=3
#p_omph = p
##OMPCV, and without CV:
#mi_mat_omph = pcu.make_mi_mat(d_omp, p_omph)
#P_omph = np.size(mi_mat_omph,0)
#Psi_omph = pcu.make_Psi(y_data[optim_indices,:d],mi_mat_omph,chc_Psi)
##    Psi_omph = pd.read_csv(f'{out_dir_ini}/plots/j={j}/Psi_omph_1dellps_n={N}_genmod_p={p_omph}_d{d_omp}_j{j}.csv').to_numpy()
##    df_Psi_omph = pd.DataFrame(Psi_omph)
##    df_Psi_omph.to_csv(f'{out_dir_ini}/plots/j={j}/Psi_omph_1dellps_n={N}_genmod_p={p_omph}_d{d_omp}_j{j}.csv',index=False)
#if chc_omp_slv=='ompcv':
#    omph = lm.OrthogonalMatchingPursuitCV(cv=5, fit_intercept=False)
#elif chc_omp_slv=='stdomp':    
#    omph = lm.OrthogonalMatchingPursuit(n_nonzero_coefs=S_omp,fit_intercept=False)
#print('omph:',omph)
##    import pdb; pdb.set_trace()
#omph.fit(Psi_omph, u_data[optim_indices].flatten())
#c_omph = omph.coef_ 
## c_ini = c_omph[:P_omp]
#S_omph = np.size(np.nonzero(c_omph)[0])
#print('S_omph:',S_omph)
#test_omp_ph, valid_omp_ph = tnn.val_test_err(data_tst,mi_mat_omph,c_omph)
#print(f'Training Error for c_omph: {test_omp_ph}')
## Validation error: is the error on the unseen data:
#print(f'Validation Error for c_omph: {valid_omp_ph}')
##import pdb;pdb.set_trace()
## Least squares:
## eps_u_tomp = la.norm(Psi[test_indices[:N_tep],:] @ c_omph - u_data[test_indices[:N_tep]])/la.norm(u_data[test_indices[:N_tep]])
#if chc_eps =='c':
#    eps_c_omp = la.norm(c_omph - c_ref)    
#    eps_c_omp_abs.append(la.norm(c_omph - c_ref))    
#    epsc_omph.append(eps_c_omp)
#df_comph = pd.DataFrame({'c_omph':c_omph})
#df_comph.to_csv(f'{out_dir_ini}/plots/j={j}/comph_1dellps_n={N}_genmod_S={S_omph}_p={p_omph}_j{j}.csv',index=False)
#df_epscomp = pd.DataFrame({'epsu_omph':valid_omp_ph,'epsu_omph_t':test_omp_ph},index=[0])
#df_epscomp.to_csv(f'{out_dir_ini}/plots/epsuomph_tst_1dellps_n={N}_genmod_S={S_omph}_j{j}.csv',index=False)

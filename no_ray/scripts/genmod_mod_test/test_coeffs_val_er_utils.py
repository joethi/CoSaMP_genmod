import numpy as np
import random
import numpy.linalg as la 
import genmod_mod_test.polynomial_chaos_utils as pcu
import genmod_mod_test.train_NN_omp_wptmg_test as tnn

def find_ls_coeff_err_usr_actset(Ph,S_lam,Lam_ls_lst,y_data,optim_indices,chc_Psi,u_data,mi_mat,data_tst,ls_ind):
    Lam_ls_ini = np.array(Lam_ls_lst)  
    Lam_chs_rng = np.setdiff1d(np.arange(0,Ph),Lam_ls_ini)
    c_ls = np.zeros(Ph)
    vler_fl_ls = []
    trer_fl_ls = []
    random.seed(ls_ind)    
    Lam_chs = random.sample(Lam_chs_rng.tolist(),int(S_lam-np.size(Lam_ls_ini))) 
    Lam_ls = np.concatenate((Lam_ls_ini,np.array(Lam_chs))) 
    #print("ls_ind",ls_ind,"Lam_ls",Lam_ls)    
    c_ls = np.zeros(Ph)
    Psi = pcu.make_Psi_drn(y_data[optim_indices,:],mi_mat,Lam_ls,chc_Psi)     
    Psi_T = np.transpose(Psi)
    c_ls_sht = (la.inv(Psi_T @ Psi) @ Psi_T @ u_data[optim_indices]).flatten()
    c_ls[Lam_ls] = c_ls_sht
    print("ls_ind",ls_ind,'c_ls[[16,23]]',c_ls[[16,23]],"Lam_ls",Lam_ls)
   # trn_err_ls, valid_err_ls = tnn.val_test_err(data_tst,mi_mat,c_ls)
    return {"ls_ind":ls_ind,"c_ls":c_ls,"Lam_ls":Lam_ls}  
   # df_cls = pd.DataFrame(c_fl_ls)
   # df_cls.to_csv(f'{out_dir_ini}/plots/c_fl_ls_1dellps_n={N}_genmod_p={p}_j{j}.csv',index=False)
   # df_epsuls = pd.DataFrame({'epsu_ls':valid_err_ls,'epsu_ls_tr':trn_err_ls},index=[0])
   # df_epsuls.to_csv(f'{out_dir_ini}/plots/epsuls_tst_1dellps_n={N}_p={p}_genmod_j{j}.csv',index=False)
#======================================================================================
def apply_lst_sqr_actset(Lam_ls,P,mi_mat,chc_Psi,u_data,y_data,optim_indices):
    c_ls = np.zeros(P)
    Psi = pcu.make_Psi_drn(y_data[optim_indices,:],mi_mat,Lam_ls,chc_Psi)
    Psi_T = np.transpose(Psi)
    c_ls_sht = (la.inv(Psi_T @ Psi) @ Psi_T @ u_data[optim_indices]).flatten()
    c_ls[Lam_ls] = c_ls_sht
    return c_ls 
#======================================================================================
#======================================================================================
#======================================================================================
    #%% Take two basis at a time and apply least squares:
    #    Lam_ls = pd.read_csv(f'{out_dir_ini}/results/csjul24_rsdl/plots/j=0/it=1/Lam_sel_1dellps_n=100_genmod_S=6_1_j0_c0.csv')['Lam_sel'].to_numpy().flatten() 
    #    Lam_ls = np.setdiff1d(Lam_ls,3145)
#    Lam_lsfl = pd.read_csv(f'{out_dir_ini}/results/csjul24_rsdl/plots/j=0/it=1/Lam_sel_1dellps_n=100_genmod_S=6_1_j0_c0.csv')['Lam_sel'].to_numpy().flatten()
#    c_fl_ls = np.zeros((P,np.size(Lam_lsfl) - 1))                               
#    vler_fl_ls = []
#    trer_fl_ls = []
#    for ls_ind in range(np.size(Lam_lsfl)-1):
#        Lam_ls = [Lam_lsfl[0],Lam_lsfl[ls_ind+1]]
#        c_ls = np.zeros(P)
#    ##    Psi_fl = pd.read_csv(f'{out_dir_ini}/plots/j={j}/Psi_omph_1dellps_n={N}_genmod_p={p}_d{d}_j{j}.csv').to_numpy()   
#        Psi = pcu.make_Psi_drn(y_data[optim_indices,:],mi_mat,Lam_ls,chc_Psi)     
#        Psi_T = np.transpose(Psi)
#        c_ls_sht = (la.inv(Psi_T @ Psi) @ Psi_T @ u_data[optim_indices]).flatten()
#        c_ls[Lam_ls] = c_ls_sht
#        c_fl_ls[:,ls_ind] = c_ls
#        print('c_ls',c_ls)
#        trn_err_ls, valid_err_ls = tnn.val_test_err(data_tst,mi_mat,c_ls)
#        vler_fl_ls.append(valid_err_ls)
#        trer_fl_ls.append(trn_err_ls)
##    res_ls = Psi_fl[:,Lam_ls] @ c_ls[Lam_ls] - u_data[optim_indices]
#    import pdb; pdb.set_trace()
##    Psi_fl = pcu.make_Psi(y_data[optim_indices,:],mi_mat,chc_Psi)
#    lam_opt = np.argmax(np.abs(Psi_fl.T @ res_ls))
#    import pdb; pdb.set_trace()
   #  df_cls = pd.DataFrame({'c_ls':c_ls})
   #  df_cls.to_csv(f'{out_dir_ini}/plots/cls_1dellps_n={N}_genmod_p={p}_j{j}.csv',index=False)
   #  df_epsuls = pd.DataFrame({'epsu_ls':valid_err_ls,'epsu_ls_tr':trn_err_ls},index=[0])
   #  df_epsuls.to_csv(f'{out_dir_ini}/plots/epsuls_tst_1dellps_n={N}_p={p}_genmod_j{j}.csv',index=False)
   #  import pdb; pdb.set_trace()
   # import pdb; pdb.set_trace()   

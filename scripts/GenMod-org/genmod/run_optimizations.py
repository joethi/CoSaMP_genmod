import sys
import numpy as np
import sklearn.linear_model as lm
from scipy.stats import sem
import pandas as pd
import numpy.linalg as la
import genmod.decay_models as dm
from genmod.utils import pickle_save
import genmod.polynomial_chaos_utils as pcu

import importlib
importlib.reload(dm)

# %%


def run_genmod(start_iter, stop_iter, fn, d, p, data_all, indices, outdir,Nt,Nv,mi_mat,nz,
               opt_params=None, lasso_iter=40000, lasso_eps=1e-10,
               lasso_tol=1e-6, lasso_n_alphas=100):
    if opt_params is None:
        opt_params = {'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8,
                      'stepSize': 0.001, 'maxIter': 100000,
                      'objecTol': 1e-6, 'ALIter': 10, 'resultCheckFreq': 5000,
                      'updateLambda': True, 'switchSigns': False,
                      'useLCurve': False, 'showLCurve': False}
    opt_params['nAlphas'] = lasso_n_alphas
    opt_params['tolLasso'] = lasso_tol
    opt_params['iterLasso'] = lasso_iter
    opt_params['epsLasso'] = lasso_eps
    Nvlrp = opt_params['Nvlrp']    
    u_data_all = data_all['u_data']
    y_data_all = data_all['y_data']
    test_err = np.zeros(stop_iter - start_iter + 1)
    valid_err = np.zeros(stop_iter - start_iter + 1)

    import pdb;pdb.set_trace()
    for j in np.linspace(start_iter, stop_iter,
                         stop_iter - start_iter + 1).astype(int):
        print('--------------------' + str(j) + '---------------------')

        # #Log results to output file
        # outfile = '/app/current_output/' + fn + '_output_genmod_' + str(j) + '.txt'
        # old_stdout = sys.stdout
        # old_stderr = sys.stderr
        # sys.stdout = open(outfile,'wt')
        # sys.stderr = sys.stdout
        min_lsvld = 0
        
        for k in range(Nvlrp):
            # Find optimization and validation indices
            optims = [name for name in indices.columns if name.startswith("optim")]
            optim_indices = indices.loc[j][optims].to_numpy()
            valids = [name for name in indices.columns if name.startswith("valid")]
            valid_indices = indices.loc[j][valids].to_numpy()
            p_z0 = np.concatenate(([np.log(np.absolute(np.mean(u_data_all)))],
                                   np.ones(nz - 1)+0.1*np.random.rand(nz - 1)))    
            # Initialize and fit decay model
            dmo = dm.DecayModel(d, p, outdir, p_z0, model='expalg')
            print("Setting data sample....")
            # dmo.set_data_sample(u_data_all[optim_indices].flatten(),
            #                     y_data_all[optim_indices, :],
            #                     u_data_all[valid_indices].flatten(),
            #                     y_data_all[valid_indices, :])
            dmo.set_data_sample(u_data_all[optim_indices].flatten(),
                                y_data_all[optim_indices, :],
                                u_data_all[valid_indices].flatten(),
                                y_data_all[valid_indices, :],u_data_all,y_data_all)            
            print("Fitting model....")
            dmo.fit_model(opt_params) 
            # Save decay model object
            # pickle_save(dmo, fn + '_' + str(j), outdir)
            if k==0:
                min_lsvld = dmo.Loss_vld[-1]
                k_min = k
                df_coef = pd.DataFrame({'Coefficients': dmo.c,
                                        'GenMod': dmo.zeta * dmo.G,
                                        'SparseVec': dmo.nu})
                # df_coef = pd.DataFrame({'Coefficients': np.abs(dmo.c)*dmo.zeta,
                #                         'GenMod': dmo.zeta * dmo.G,
                #                         'SparseVec': dmo.nu})
                df_coef.to_csv(f'{outdir}/{fn}_genmod_kmin={k}_{j}.csv', index=False)
                df_prms0 = pd.DataFrame({'prms_fl': dmo.z_str})
                df_prms0.to_csv(f'{outdir}/{fn}_genmod_z0_kmin={k}_{j}.csv', index=False)
                # df_prms_fl = pd.DataFrame({'prms_end': dmo.z})
                # df_prms_fl.to_csv(f'{outdir}/{fn}_genmod_zfl_kmin={k}_{j}.csv', index=False)                
                df_cz1 = pd.DataFrame({'epoch':dmo.epc_plt,'Loss':dmo.Loss_opt,'Loss_vld':dmo.Loss_vld})
                df_cz1.to_csv(f'{outdir}/errplt_N={Nt}_kmin={k}_{j}.csv', index=False)
            if dmo.Loss_vld[-1] < min_lsvld:
                min_lsvld = dmo.Loss_vld[-1]
                k_min = k
                df_coef = pd.DataFrame({'Coefficients': dmo.c,
                                        'GenMod': dmo.zeta * dmo.G,
                                        'SparseVec': dmo.nu})
                # df_coef = pd.DataFrame({'Coefficients': np.abs(dmo.c)*dmo.zeta,
                #                         'GenMod': dmo.zeta * dmo.G,
                #                         'SparseVec': dmo.nu})
                df_coef.to_csv(f'{outdir}/{fn}_genmod_kmin={k}_{j}.csv', index=False)
                df_prms0 = pd.DataFrame({'prms_fl': dmo.z_str})
                df_prms0.to_csv(f'{outdir}/{fn}_genmod_z0_kmin={k}_{j}.csv', index=False)                
                df_cz1 = pd.DataFrame({'epoch':dmo.epc_plt,'Loss':dmo.Loss_opt,'Loss_vld':dmo.Loss_vld})
                df_cz1.to_csv(f'{outdir}/errplt_N={Nt}_kmin={k}_{j}.csv', index=False)
            # Save coefficients
            # df_coef = pd.DataFrame({'Coefficients': dmo.c,
            #                         'GenMod': dmo.zeta * dmo.G,
            #                         'SparseVec': dmo.nu})
            # df_coef.to_csv(f'{outdir}/{fn}_genmod_k={k}_{j}.csv', index=False)
            # df_cz1 = pd.DataFrame({'epoch':dmo.epc_plt,'Loss':dmo.Loss_opt,'Loss_vld':dmo.Loss_vld})
            # df_cz1.to_csv(f'{outdir}/errplt_N={Nt}_k={k}_{j}.csv', index=False)
        # Write the final updated parameters:
        df_prms = pd.DataFrame({'prms_fl': dmo.z})
        df_prms.to_csv(f'{outdir}/{fn}_genmod_zfl_{j}.csv', index=False)
        # #Reset where output and error messages are sent
        # sys.stdout = old_stdout
        # sys.stderr = old_stderr
        cf = df_coef['Coefficients'].to_numpy()
        optim_indices = indices.iloc[j].to_numpy()
        valid_indices = np.setdiff1d(range(np.size(u_data_all)), optim_indices)

        valids = [name for name in indices.columns if name.startswith("valid")]
        test_indices = indices.loc[j][valids].to_numpy()

        # Testing error
        #===================================================================
        # outdir1 = 'C:/Users/jothi/OneDrive - UCB-O365/PhD/UQ_research/ACCESS_UQ/GenMod-NN/GenMod-mod/output/1DElliptic/N=4k/genmod_Psi_tot.csv'
        # Psi_tot = pd.read_csv(outdir1)
        # op_ind1 = test_indices.tolist()
        # Psi_test = Psi_tot.loc[op_ind1,:].to_numpy()
        Psi_test = pcu.make_Psi(y_data_all[test_indices, :d], mi_mat)
        test_err[j] = la.norm(
            Psi_test @ cf - u_data_all[test_indices].T
        ) / la.norm(u_data_all[test_indices].T)
        # Validation error
        # val_ind1 = valid_indices.tolist()
        # Psi_valid = Psi_tot.loc[val_ind1[:500],:].to_numpy()
        Psi_valid = pcu.make_Psi(y_data_all[valid_indices[:Nv], :d], mi_mat)
        valid_err[j] = la.norm(
            Psi_valid @ cf - u_data_all[valid_indices[:Nv]].T
        ) / la.norm(u_data_all[valid_indices[:Nv]].T)    
        debug = 0 
    df_err= pd.DataFrame({'vl_err': valid_err,'tst_err': test_err})
    df_err.to_csv(f'{outdir}/{fn}_genmod_err_{Nt}.csv')
    print('Valid error',valid_err)
    print('Test error',test_err)       
#
def run_l1(start_iter, stop_iter, fn, d, p, data_all, indices, outdir,
           lasso_iter=40000, lasso_eps=1e-10, lasso_tol=1e-6,
           lasso_n_alphas=100, t_iterations=3, runOMP=True, runLasso=True,
           plot_L_curve=True):

    u_data_all = data_all['u_data']
    y_data_all = data_all['y_data']

    mi_mat = pcu.make_mi_mat(d, p)

    for i in np.arange(start_iter, stop_iter + 1):
        print(i)

        indices_i = indices.loc[i].to_numpy()

        # Use all data for the l1 fit
        Psi = pcu.make_Psi(y_data_all[indices_i], mi_mat)
        u = u_data_all[indices_i]

        # OMP
        if runOMP:
            print('Running OMP...')
            omp = lm.OrthogonalMatchingPursuitCV(cv=5, fit_intercept=False)
            omp.fit(Psi, u)

            # Save coefficients
            df_coef = pd.DataFrame({'Coefficients': omp.coef_})
            df_coef.to_csv(f'{outdir}/{fn}_omp_{i}.csv',
                           index=False)

        # IRW Lasso
        if runLasso:
            # outfile = f'/app/current_output/{fn}_output_IRW_Lasso_{i}.txt'
            # old_stdout = sys.stdout
            # old_stderr = sys.stderr
            # sys.stdout = open(outfile,'wt')
            # sys.stderr = sys.stdout

            print('Running IRW-Lasso...')
            epsilon = 1e-4
            iter = 0
            while True:
                if iter == 0:
                    W_inv = np.eye(np.size(Psi, 1))
                else:
                    W = np.diag(1 / (np.absolute(W_inv @ las2.coef_) + epsilon))
                    W_inv = np.linalg.inv(W)

                las = lm.LassoCV(cv=5, n_alphas=lasso_n_alphas, eps=lasso_eps,
                                 tol=lasso_tol, max_iter=lasso_iter,
                                 fit_intercept=False)
                las.fit(Psi @ W_inv, u)

                # Pick alpha using "one-standard_error" rule
                mean_path = np.mean(las.mse_path_, 1)
                sem_path = sem(las.mse_path_, 1)
                min_index = np.where(mean_path == np.min(mean_path))
                min_index = list(min_index)[0][0]
                mean_plus_sem = mean_path[min_index] + sem_path[min_index]
                for index in np.linspace(min_index - 1, 0,
                                         min_index).astype(int):
                    if mean_path[index] > mean_plus_sem:
                        break
                alpha = las.alphas_[index + 1]

                # Rerun lasso with the alpha determined above
                las2 = lm.Lasso(alpha=alpha, fit_intercept=False,
                                tol=lasso_tol, max_iter=lasso_iter)
                las2.fit(Psi @ W_inv, u)

                # Store coefficients from standard lasso
                # (i.e., only one iteration of IRW-Lasso)
                if iter == 0:
                    lasso_coef = np.copy(las2.coef_)

                # Check for convergence
                if iter > 0:
                    coef_diff = np.linalg.norm(W_inv @ las2.coef_ - previous_coef) / np.linalg.norm(previous_coef)
                    print('Coef diff: ' + str(coef_diff))
                    if coef_diff < 1e-4:
                        print('Converged')
                        break
                previous_coef = np.copy(W_inv @ las2.coef_)

                # Increase epsilon if did not converged in 20 iterations
                if (iter + 1) % 20 == 0:
                    epsilon = epsilon * 10
                    print(f'Did not converge. Increasing epsilon to {epsilon}')
                if epsilon > 1:
                    print('Epsilon > 1: We are breaking.')
                    break

                iter += 1

            # Print results to log file
            print(str.format('....Alpha: {:g}', las.alpha_))
            print(str.format('....Alpha (with SEM): {:g}', alpha))
            print(str.format('....Max Alpha: {:g}', np.max(las.alphas_)))
            print(str.format('....Min Alpha: {:g}', np.min(las.alphas_)))

            # Save coefficients
            df_coef = pd.DataFrame({
                'IRW-Lasso-Coefficients': W_inv @ las2.coef_,
                'Lasso-Coefficients': lasso_coef
            })
            df_coef.to_csv(f'{outdir}/{fn}_las_{i}.csv',
                           index=False)

            # #Reset where output and error messages are sent
            # sys.stdout = old_stdout
            # sys.stderr = old_stderr

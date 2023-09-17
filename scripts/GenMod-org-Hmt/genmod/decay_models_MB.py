import time

from scipy.special import factorial
import numpy as np
import random
import sklearn.linear_model as lm
from scipy.stats import sem
import scipy.special as sps
import genmod.polynomial_chaos_utils as pcu
import genmod.decay_model_utils as dmu
import genmod.L_curve_utils_lasso as lcu
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)

class DecayModel:

    def __init__(self, d, p, outdir,prm_0,N, model='expalg'):

        # Constant paramters
        self.d = d
        self.p = p
        self.P = 1
        for i in range(p):
            self.P = self.P * (p - i + d)
        self.P = int(self.P / sps.factorial(p))
        self.mi_mat = pcu.make_mi_mat(d, p)
        self.mi_c = dmu.make_multiindex_constant(self.mi_mat)
        self.beta_param_mat = dmu.make_beta_param_mat(d, self.mi_mat,
                                                      model=model)
        self.k = np.size(self.beta_param_mat, 1)
        self.outdir1 = outdir 
        self.prm_0 = prm_0
        self.N_opt = int(4*N/5)
        # Lasso parameters
        self.alpha = 0

    def update_z(self, z, add_to_hist):
        """Update the value of z and parameters that depend on z."""
        # Make sure that there is exponential decay (if growing set to zero)
        z_exp = z[1:1 + self.d]
        z_exp[z_exp < 0] = 0
        z[1:1 + self.d] = z_exp

        # Update z and G(z)
        self.z = z
        self.G = dmu.decay_model(z, self.mi_c, self.beta_param_mat)
        self.c = self.nu + self.G * self.zeta

        if add_to_hist:
            self.zdict[self.iter] = np.vstack((self.zdict[self.iter], self.z))

    def update_nu(self, nu):
        """Update the value of nu and the history."""
        self.nu = nu
        self.nuhist = np.vstack((self.nuhist, self.nu))
        self.c = self.nu + self.G * self.zeta

    def update_zeta(self, zeta, add_to_hist=True):
        """Set the value of signs within the vector zeta."""
        self.zeta = zeta
        self.c = self.nu + self.G * self.zeta

        if add_to_hist:
            self.zeta_hist = np.vstack((self.zeta_hist, zeta))

    def set_data_sample_random(self, data_tot, N_sample, N_valid):
        u_data_all = data_tot['u_data']
        y_data_all = data_tot['y_data']
        N_tot = np.size(u_data_all)

        # Pick random sample indices
        self.sample_indices = random.sample(range(N_tot), N_sample)
        not_sample_indices = np.setdiff1d(range(N_tot), self.sample_indices)
        self.valid_indices = random.sample(list(not_sample_indices), N_valid)

        # Set data variables
        self.set_data_sample(u_data_all[self.sample_indices].flatten(),
                             y_data_all[self.sample_indices, :],
                             u_data_all[self.valid_indices].flatten(),
                             y_data_all[self.valid_indices, :])

    def set_data_sample(self, u_data, y_data, u_valid, y_valid, z0=None):
        # Set Adam training data
        self.u_data = u_data
        self.y_data = y_data
        self.Psi = pcu.make_Psi(self.y_data, self.mi_mat)

        # Set Adam validation data
        self.u_valid = u_valid
        self.y_valid = y_valid
        self.Psi_valid = pcu.make_Psi(self.y_valid, self.mi_mat)

        # Generate signs with OMP using all data
        u = np.concatenate((self.u_data, self.u_valid))
        Psi = np.vstack((self.Psi, self.Psi_valid))
        self.Psi_tot = Psi
        # Write data
        # df_psi_tot = pd.DataFrame(self.Psi_tot)
        # df_psi_tot.to_csv(f'{self.outdir1}/genmod_Psi_tot.csv', index=False)
        zeta = dmu.set_omp_signs(Psi, u)
        # Read zeta from the trained data to set right signs:
        # outdir1 = f'{self.outdir1}/N=40k_trn/genmod_csign.csv'
        # zeta1 = pd.read_csv(outdir1)
        # zeta = zeta1['sign_c'].to_numpy()
        # Set initial conditions using mean of data and assume decay
        z0 = self.prm_0 
        # z0 = np.concatenate(([np.log(np.absolute(np.mean(u)))],
                              # np.ones(self.k - 1)+0.1*np.random.rand(self.k - 1)))
        # z0 = np.random.rand(self.k)
        # df_z0 = pd.DataFrame({'z0':z0})
        # df_z0.to_csv(f'{self.outdir1}/genmod_z0_rnd.csv', index=False)
        self.z_str = z0 
        # Add noise to z0:
        # z0 = z0 + 0.5 * np.random.rand(self.k)    
        self.set_initial_conditions(z0, zeta=zeta)
        # For plotting:
        # df_zstr = pd.read_csv(f'{self.outdir1}/N=40k_trn/genmod_z0_rnd.csv')        
        # self.z_str = df_zstr['z0'].to_numpy()            
        # df_cstr = pd.read_csv(f'{self.outdir1}/N=40k_trn/genmod_cs_init.csv')        
        # self.c_str = df_cstr['c'].to_numpy()    
        
    def set_initial_conditions(self, z0, nu0=None, zeta=None):
        # nu
        if nu0 is None:
            nu0 = np.zeros(self.P)
        self.nuhist = nu0
        self.nu = nu0

        # zeta
        if zeta is None:
            zeta = np.ones(self.P)
        self.zeta = zeta
        self.zeta_hist = np.empty((0, np.size(zeta)))

        # z and G
        self.zdict = {}
        self.update_z(z0, False)
        ## write data for verification:
        #================Uncomment to write u_new_data:
        # u_ver = self.Psi_tot @ self.c
        # df_u = pd.DataFrame({'u_ver': u_ver})
        # df_u.to_csv(f'{self.outdir1}/genmod_u_ver.csv', index=False)
        # df_cini = pd.DataFrame({'c': self.c})
        # df_cini.to_csv(f'{self.outdir1}/genmod_cs_init.csv', index=False)
        # df_csgn = pd.DataFrame({'sign_c': np.sign(self.c)})
        # df_csgn.to_csv(f'{self.outdir1}/genmod_csign.csv', index=False)        
        self.update_zeta(zeta, False)

    def fit_model(self, opt_params, batch_size=None):

        self.opt_params = opt_params
        batch_size = self.opt_params['batch_size']
        # Set initial objective_change
        self.update_objective()

        # Alternate Adam and Lasso
        for i in range(self.opt_params['ALIter']):

            self.iter = 'Iter:' + str(i)
            self.zdict[self.iter] = np.empty([0, np.size(self.z)])

            # First document initial condition
            self.update_z(self.z, True)
            self.update_zeta(self.zeta)

            # Run Adam optimization (update z)
            print('Running Adams: ' + str(i))
            self.adam_optim(opt_params['maxIter'], opt_params['objecTol'],
                            batch_size,i)
            self.update_objective()

            # Dont rerun lasso if objective hasn't changed
            if i > 0:
                if self.objt_valid > objt_after_nu - 1e-8:
                    # df_cz = pd.DataFrame({'epoch':self.epc_plt,'Loss':self.Loss_opt,'Loss_vld':self.Loss_vld})
                    # df_cz.to_csv(f'{self.outdir1}/errplt_{i}.csv', index=False)            
                    print('Adam did not improve validation objective. Breaking...')
                    break

            # Next run weighted Lasso using all the data
            print(f'Running Weighted Lasso: {i}')
            print(f'Starting Validation Objective: {self.objt_valid}')
            objt_before_nu = self.objt_valid
            nu_las = self.run_lasso()

            # Only continue and update nu if objective has improved
            diff = self.Psi_valid @ (self.zeta * self.G + nu_las) - self.u_valid
            objt_after_nu = np.linalg.norm(diff)**2
            if objt_after_nu - self.objt_valid > 0:
                print('Validation error new: ' + str(objt_after_nu))
                print('Lasso did not improve validation objective. Breaking...')
                break

            self.update_nu(nu_las)
            self.update_objective()
            objt_after_nu = self.objt_valid
            print(str.format('Ending Validation Objective: {:g}', self.objt_valid))

            # Check stopping criteria
            if i > 0:
                objt_best = np.minimum(objt_before_nu, self.objt_valid)
                print(f'Best Validation Objective: {objt_best}')
                print(f'Previous Best Validation Objective: {objt_old}')
                print(str((objt_best - objt_old) / objt_old))
                if (objt_best - objt_old) / objt_old > -10**-6:
                    print('Final stopping criteria met.')
                    break

            objt_old = np.minimum(objt_before_nu, self.objt_valid)

        # Update with final objective
        self.iter = f'Iter: {i+1}'
        self.zdict[self.iter] = np.empty([0, np.size(self.z)])
        self.update_z(self.z, True)
        print(f'Final Validation Objective: {self.objt_valid}')

    def update_objective(self):
        diff = self.Psi_valid @ self.c - self.u_valid
        self.objt_valid = np.linalg.norm(diff)**2

    # Functions for running Adam optimization

    def adam_optim(self, maxIter, objecTol, batch_size,plt_ind):
        m = np.zeros(np.size(self.z))
        v = np.zeros(np.size(self.z))
        t = 1
        objt_adam_new = 10**8

        # Set initial best z
        objt_valid_best = self.objt_valid
        z_best = self.z

        # Set deriv() function to be used later
        if batch_size is None:
            deriv = self.adam_deriv
        else:
            self.batch_size = batch_size
            # deriv = self.adam_deriv_batches

        best_iter = 0
        if plt_ind==0:
            # zer_str = []
            # cer_str = []
            self.epc_plt = []
            self.Loss_opt = []
            self.Loss_vld = []
            self.ep_ind = -1
        for k in range(maxIter):
            self.ep_ind += 1
            btch_iter = int((self.N_opt/self.batch_size))
            rnd_indc = random.sample(range(self.N_opt),self.N_opt)
            # Print results at specified intervals and check convergence
            for mnb in range(btch_iter):
                if mnb==0:
                    rnd_btch = rnd_indc[0:self.batch_size]
                else:    
                    rnd_btch = rnd_indc[mnb*self.batch_size:(mnb+1)*self.batch_size]
                deriv = self.adam_deriv_batches
                if k > 0 and k % self.opt_params['resultCheckFreq'] == 0:
                    objec_conv = self.check_and_print_results(k, objt_adam,
                                                              objt_adam_new)
                    # z_err = np.linalg.norm(self.z_str - self.z) /np.linalg.norm(self.z_str)
                    # zer_str.append(z_err)
                    # c_err = np.linalg.norm(self.c_str - self.c) /np.linalg.norm(self.c_str)
                    # cer_str.append(c_err)
                    self.epc_plt.append(self.ep_ind)
                    self.Loss_opt.append(objt_adam_new)
                    self.Loss_vld.append(self.objt_valid)
                    # plt.semilogy(k,z_err,'k*')
                    # plt.semilogy(k,c_err,'ro')
                    if objec_conv:
                        break
    
                # Perform Adam iteration with or without batches
                objt_adam = objt_adam_new
                diff, objt_deriv = deriv(rnd_btch)
                objt_adam_new = np.linalg.norm(diff)**2
                z, m, v = self.adam_update(objt_deriv, m, v, t)
                self.update_z(z, True)
                t += 1
    
                # See how fit looks on validation data,
                self.update_objective()
                if self.objt_valid < objt_valid_best:
                    best_iter = k
                    objt_valid_best = self.objt_valid
                    z_best = self.z
    
                if k > 20000:
                    if k - best_iter > 100:
                        print('Best validation iteration occured 100 iterations ago. Breaking...')
                        break

        # Use best value of z found but don't add to optimization history
        print(f'Best iteration for validation data: {best_iter}')
        self.update_z(z_best, False)
        # plotting stuff:
        # plt.legend(['$z^{*}_{err}$','$c^{*}_{err}$'])
        # plt.xlabel('epochs')
        # plt.ylabel('$||x^{*}-x_{i}||/||x^{*}||$')
        # df_cz = pd.DataFrame({'epoch':epc_plt,'z_err': zer_str,'c_err': cer_str,'Loss':Loss_opt,'Loss_vld':Loss_vld})
        # df_cz = pd.DataFrame({'epoch':epc_plt,'Loss':Loss_opt,'Loss_vld':Loss_vld})
        # df_cz.to_csv(f'{self.outdir1}/errplt.csv', index=False)            

    def adam_deriv(self):
        diff = self.Psi @ self.c - self.u_data
        obj_deriv = self.find_objective_derivative(self.Psi, diff)
        return(diff, obj_deriv)

    def adam_deriv_batches(self,batch_indices):
        # batch_indices = random.sample(range(np.size(self.u_data)),
                                      # self.batch_size)
        diff = self.Psi[batch_indices, :] @ self.c - self.u_data[batch_indices]
        obj_deriv = self.find_objective_derivative(self.Psi[batch_indices, :],
                                                   diff)
        return(diff, obj_deriv)

    def find_objective_derivative(self, Psi, diff):
        G_Jac =  dmu.decay_model_jacobian(self.G, self.beta_param_mat) * self.zeta[:, np.newaxis]
        obj_deriv = 2 * diff @ Psi @ G_Jac
        return(obj_deriv)

    def adam_update(self, obj_deriv, m, v, t):
        beta1 = self.opt_params['beta1']
        beta2 = self.opt_params['beta2']
        epsilon = self.opt_params['epsilon']
        stepSize = self.opt_params['stepSize']
        m = beta1 * m + (1 - beta1) * obj_deriv
        v = beta2 * v + (1 - beta2) * obj_deriv**2
        mHat = m / (1 - beta1**t)
        vHat = v / (1 - beta2**t)
        z_change = -stepSize * mHat / (np.sqrt(vHat) + epsilon)
        z = self.z + z_change
        return z, m, v

    def run_lasso(self):
        Psi = np.vstack((self.Psi, self.Psi_valid))
        u = np.concatenate((self.u_data, self.u_valid))

        # Calculate sparse vector using weighted lasso
        weight_vec_inv = self.G + 10**-4 * self.G[0]  # 10**-5*self.G[1]
        weight_mat_inv = np.diag(weight_vec_inv)

        if self.opt_params['useLCurve']:
            alpha_final, x_final, y_final = lcu.findAlphaMaxCurve(
                Psi @ weight_mat_inv, u - Psi @ (self.zeta * self.G), 1e-20,
                "L curve", plot_results=self.opt_params['showLCurve'],
                a_num=100, d_lim=0.001
            )
            las = lm.Lasso(alpha=alpha_final, fit_intercept=False,
                           tol=1e-8, max_iter=100000)
            las.fit(Psi @ weight_mat_inv, u - Psi @ (self.zeta * self.G))
        else:
            las1 = lm.LassoCV(cv=5, n_alphas=self.opt_params['nAlphas'],
                              eps=self.opt_params['epsLasso'],
                              tol=self.opt_params['tolLasso'],
                              max_iter=self.opt_params['iterLasso'],
                              fit_intercept=False)
            las1.fit(Psi @ weight_mat_inv, u - Psi @ (self.zeta * self.G))

            # Pick alpha using "one-standard_error" rule
            mean_path = np.mean(las1.mse_path_, 1)
            sem_path = sem(las1.mse_path_, 1)
            min_index = np.where(mean_path == np.min(mean_path))
            min_index = list(min_index)[0][0]
            mean_plus_sem = mean_path[min_index] + sem_path[min_index]
            final_index = min_index
            for index in np.linspace(min_index - 1, 0, min_index).astype(int):
                if mean_path[index] > mean_plus_sem:
                    break
                final_index = index + 1
            alpha = las1.alphas_[final_index]
            self.alpha = alpha

            las = lm.Lasso(alpha=alpha, fit_intercept=False,
                           tol=self.opt_params['tolLasso'],
                           max_iter=self.opt_params['iterLasso'])
            las.fit(Psi @ weight_mat_inv, u - Psi @ (self.zeta * self.G))

            # self.alpha = las1.alpha_
            self.print_lasso_results(las1)

        nu_las = weight_mat_inv @ las.coef_

        # Change sign if needed
        signs = self.zeta
        if self.opt_params['switchSigns']:
            signChange = np.sign(self.zeta * self.G + nu_las) * signs == -1
            print(f'Flipping {np.size(np.where(signChange))} signs.')

            # Flip sign and modify the nu vector
            signs[signChange] = signs[signChange] * -1
            nu_las[signChange] = 2 * self.zeta[signChange] * self.G[signChange] + nu_las[signChange]

        # Update signs
        self.update_zeta(signs)

        # Print the nu coefficients
        print(nu_las[np.absolute(nu_las) > 0])

        return(nu_las)

    def find_c(self, nu, zeta, z):
        return(nu + zeta * dmu.decay_model(z, self.mi_c, self.beta_param_mat))

    # Functions for checking for convergence and printing results

    def check_and_print_results(self, k, obj, obj_new):
        print(str.format('Iteration: {:g}', k))
        objective_change = np.absolute(obj - obj_new)
        objective_change_percent = objective_change / obj

        print(str.format('Objective: {:g}', obj_new))
        print(str.format('Validation Objective: {:g}', self.objt_valid))
        print(str.format('Objective Change Percent: {:g}', objective_change_percent))
        if objective_change_percent < self.opt_params['objecTol']:
            return(True)
        return(False)

    def print_lasso_results(self, las):
        print(str.format('Alpha: {:g}', las.alpha_))
        print(str.format('Alpha (with SEM): {:g}', self.alpha))
        print(str.format('Max Alpha: {:g}', np.max(las.alphas_)))
        print(str.format('Min Alpha: {:g}', np.min(las.alphas_)))

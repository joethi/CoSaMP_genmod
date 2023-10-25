import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
import pickle
import random
import sys

import genmodpce.polynomial_chaos_utils as pcu
import genmodpce.decay_model_utils as dmu
from . import decay_models

def plot_coef_comparison(coefs_fitted,c_omp,c_las,omp_sign,c_ls,fit_label,dataset):

    xrange = np.linspace(1,np.size(c_ls),np.size(c_ls))

    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure(figsize=(5.5,6))
    ax1 = plt.subplot(211)
    plt.grid(True)
    plt.scatter(xrange,np.absolute(c_ls),facecolor='gray',label="Least Squares",marker='s')

    plt.scatter(xrange,np.absolute(coefs_fitted),label="GenMod",marker='o',alpha=.7)
    plt.scatter(xrange,np.absolute(c_omp),label="OMP",marker='s',alpha=.7)
    plt.scatter(xrange,np.absolute(c_las),label="Lasso",marker='^',alpha=.7)

    plt.loglog()
    plt.xlim([.9,np.max(xrange)])
    plt.ylim([10**-9,np.max(np.concatenate([c_ls,coefs_fitted,c_omp,c_las]))*4])
    plt.ylabel('Coefficient magnitude')
    plt.legend()


    label_vec = (np.sign(coefs_fitted)*np.sign(c_ls)).astype(int)
    ci = label_vec==1 #correct sign index
    ii = label_vec==-1 #incorrectsing index

    max_incorrect = np.max(c_ls[ii])
    omp_vec = (np.sign(coefs_fitted)*omp_sign).astype(int)
    fi = omp_vec==-1 #coefficients whose sign flipped

    plt.subplot(212, sharex=ax1)
    plt.grid(True)
    plt.scatter(xrange,np.absolute(c_ls),facecolor='gray',label="Least Squares",marker='s',s=3)

    plt.scatter(xrange[ci],np.absolute(coefs_fitted[ci]),
        facecolor='none',edgecolor='#5DA5DA')
    plt.scatter(xrange[ci],np.absolute(coefs_fitted[ci]),
        facecolor=mpl.colors.to_rgba('#5DA5DA',.6),edgecolor='#5DA5DA',
        label="GenMod: Correct Sign") #For some reason this command only adds outline to legend markers.

    plt.scatter(xrange[ii],np.absolute(coefs_fitted[ii]),
        facecolor='none',edgecolor='#F15854')
    plt.scatter(xrange[ii],np.absolute(coefs_fitted[ii]),
        facecolor=mpl.colors.to_rgba('#F15854',.6),edgecolor='#F15854',
        label="GenMod: Incorrect Sign") #For some reason this command only adds outline to legend markers.



    plt.scatter(xrange[fi & ci],np.absolute(coefs_fitted[fi & ci]),marker='*',edgecolor='black',facecolor=cols[0],
        s=35,label='Sign flipped (correctly)')
    plt.scatter(xrange[fi & ii],np.absolute(coefs_fitted[fi & ii]),marker='*',edgecolor='black',facecolor=cols[-2],
            s=35,label='Sign flipped (incorrectly)')

    plt.loglog()
    plt.xlim([.9,np.max(xrange)])
    plt.ylim([10**-9,np.max(np.concatenate([c_ls,coefs_fitted]))*4])
    plt.ylabel('Coefficient magnitude')
    plt.xlabel('Index of PC coefficient')
    plt.legend(ncol=2)

    plt.savefig('/app/current_output/fig_data' + dataset + '_coef_mag.pdf')
    plt.show()

    return(np.sum(fi & ci),np.sum(fi & ii),max_incorrect)

def plot_coef_and_recons(N_sample_sets,fn,dataset,u_data,y_data,n_valid,bin_num=20,plotTrain=False,plotNoSparse=False,plotRecons=True,plotCoefs=True,c_ls = None):

    N_tot = np.size(u_data)
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

    #Initialize vectors to contain error
    if plotCoefs:
        model_coef_err = np.zeros(N_sample_sets)
        las_coef_err = np.zeros(N_sample_sets)
        omp_coef_err = np.zeros(N_sample_sets)
        coef_err_diff = np.zeros(N_sample_sets)

    if plotTrain:
        model_train_err = np.zeros(N_sample_sets)
        las_train_err = np.zeros(N_sample_sets)
        omp_train_err = np.zeros(N_sample_sets)
        train_err_diff = np.zeros(N_sample_sets)
        if plotNoSparse:
            model2_train_err = np.zeros(N_sample_sets)

    if plotRecons:
        model_recons_err = np.zeros(N_sample_sets)
        las_recons_err = np.zeros(N_sample_sets)
        omp_recons_err = np.zeros(N_sample_sets)
        recons_err_diff = np.zeros(N_sample_sets)
        if plotNoSparse:
            model2_recons_err = np.zeros(N_sample_sets)

    for i in range(N_sample_sets):
        dmo = pickle.load(open(fn + '_' + str(i) + '.pkl','rb'))
        omp = pickle.load(open(fn + '_omp_' + str(i) + '.pkl','rb'))
        las = pickle.load(open(fn + '_las_' + str(i) + '.pkl','rb'))

        #Validation data (data not used in optimization)
        data_indices = np.concatenate((dmo.sample_indices,dmo.valid_indices))
        not_data_indices = np.setdiff1d(range(N_tot),data_indices)
        valid_data_indices = random.sample(list(not_data_indices),n_valid)
        u_valid = u_data[valid_data_indices]
        y_valid = y_data[valid_data_indices]
        Psi = pcu.make_Psi(y_valid,dmo.mi_mat)

        u_train = np.concatenate((dmo.u_data,dmo.u_valid))
        Psi_train = np.vstack((dmo.Psi,dmo.Psi_valid))

        #OMP
        c_omp = omp.coef_

        #Lasso
        c_las = las.coef_

        if plotCoefs:
            model_coef_err[i] = np.linalg.norm(dmo.c-c_ls)/np.linalg.norm(c_ls)
            omp_coef_err[i] = np.linalg.norm(c_omp-c_ls)/np.linalg.norm(c_ls)
            las_coef_err[i] = np.linalg.norm(c_las-c_ls)/np.linalg.norm(c_ls)
            coef_err_diff[i] = (omp_coef_err[i] - model_coef_err[i])/omp_coef_err[i]

        if plotTrain:
            model_train_err[i] = np.linalg.norm(Psi_train @ dmo.c - u_train)/np.linalg.norm(u_train)
            omp_train_err[i] = np.linalg.norm(Psi_train @ c_omp - u_train)/np.linalg.norm(u_train)
            las_train_err[i] = np.linalg.norm(Psi_train @ c_las - u_train)/np.linalg.norm(u_train)
            train_err_diff[i] = (omp_train_err[i] - model_train_err[i])/omp_train_err[i]

            if plotNoSparse:
                z = dmo.zdict['Iter:0'][-1,:]
                c = dmo.zeta*dmu.decay_model(z,dmo.mi_c,dmo.beta_param_mat)
                model2_train_err[i] = np.linalg.norm(Psi_train @ c - u_train)/np.linalg.norm(u_train)

        if plotRecons:
            model_recons_err[i] = np.linalg.norm(Psi @ dmo.c - u_valid)/np.linalg.norm(u_valid)
            omp_recons_err[i] = np.linalg.norm(Psi @ c_omp - u_valid)/np.linalg.norm(u_valid)
            las_recons_err[i] = np.linalg.norm(Psi @ c_las - u_valid)/np.linalg.norm(u_valid)
            recons_err_diff[i] = (omp_recons_err[i] - model_recons_err[i])/omp_recons_err[i]

            if plotNoSparse:
                z = dmo.zdict['Iter:0'][-1,:]
                c = dmo.zeta*dmu.decay_model(z,dmo.mi_c,dmo.beta_param_mat)
                model2_recons_err[i] = np.linalg.norm(Psi @ c - u_valid)/np.linalg.norm(u_valid)

    #Initialize figure
    fig = plt.figure(figsize=(6,7))

    if plotCoefs:
        coef_err_min = np.min(np.concatenate((model_coef_err,omp_coef_err,las_coef_err)))
        coef_err_max = np.max(np.concatenate((model_coef_err,omp_coef_err,las_coef_err)))

        #PLOT: Compare coefficient error
        ax1 = plt.subplot(321)
        bins=np.histogram(np.hstack((model_coef_err,omp_coef_err,las_coef_err)),bins=bin_num)[1]
        binwidth = bins[1]-bins[0]
        n1, _ , _ = plt.hist(model_coef_err, facecolor = mpl.colors.to_rgba(cols[0],.7), edgecolor=cols[0], label='GenMod', bins=bins)
        n2, _ , _ = plt.hist(omp_coef_err, facecolor = mpl.colors.to_rgba(cols[1],.5), edgecolor=cols[1], label='OMP', bins=bins+binwidth/3)
        n3, _ , _ = plt.hist(las_coef_err, facecolor = mpl.colors.to_rgba(cols[2],.3), edgecolor=cols[2],  label='Lasso', bins=bins+2*binwidth/3)

        plt.xlim([0,bins[-1]*1.1])
        plt.xlabel(r'Relative coefficient error ($\varepsilon_c$)')
        plt.ylabel('Frequency')
        plt.legend()

        #PLOT: Difference between coefficient error
        ax2 = plt.subplot(322)
        ax2.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        n, _, _ = plt.hist(100*coef_err_diff,bins=bin_num, facecolor = mpl.colors.to_rgba(cols[-1],.5), edgecolor=cols[-1])
        plt.xlabel(r'Percent Improvement')

        #Print number of replications that didn't improve
        print(str.format('Numbr of Delta c < 0: {:g}',np.size(np.where(coef_err_diff<0))))

    if plotRecons:

        recons_err_min = np.min(np.concatenate((model_recons_err,omp_recons_err,las_recons_err)))
        recons_err_max = np.max(np.concatenate((model_recons_err,omp_recons_err,las_recons_err)))

        #PLOT: Compare reconstruction error
        ax3 = plt.subplot(323)
        bins=np.histogram(np.hstack((model_recons_err,omp_recons_err,las_recons_err)),bins=bin_num)[1]
        binwidth = bins[1]-bins[0]
        bins=np.concatenate((bins,[bins[-1]+binwidth]))
        n1, _ , _ = plt.hist(model_recons_err, facecolor = mpl.colors.to_rgba(cols[0],.7), edgecolor=cols[0], label='GenMod', bins=bins)
        n2, _ , _ = plt.hist(omp_recons_err, facecolor = mpl.colors.to_rgba(cols[1],.5), edgecolor=cols[1], label='OMP', bins=bins-binwidth/3)
        n3, _ , _ = plt.hist(las_recons_err, facecolor = mpl.colors.to_rgba(cols[2],.3), edgecolor=cols[2],  label='Lasso', bins=bins-2*binwidth/3)
        if plotNoSparse:
            n4, _ , _ = plt.hist(model2_recons_err, facecolor = mpl.colors.to_rgba(cols[3],.3), edgecolor=cols[3],  label='GenMod-NoSparse', bins=bins-2*binwidth/3)
            n3 = np.concatenate((n3,n4))
        max_height = np.max(np.concatenate((n1,n2,n3)))

        plt.xlim([0,bins[-1]*1.1])
        plt.xlabel(r'Relative reconstruction error')
        plt.ylabel('Frequency')
        plt.legend()

        #PLOT: Difference between reconstruction error
        ax4 = plt.subplot(324)
        ax4.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        n, _, _ = plt.hist(100*recons_err_diff, bins=bin_num,
            facecolor = mpl.colors.to_rgba(cols[-1],.5),
            edgecolor=cols[-1],
            label="From OMP to GenMod")
        plt.xlabel(r'Percent Improvement')
        plt.legend()
        print(str.format('Number of Delta u < 0: {:g}',np.size(np.where(recons_err_diff<0))))


    if plotTrain:

        train_err_min = np.min(np.concatenate((model_train_err,omp_train_err,las_train_err)))
        train_err_max = np.max(np.concatenate((model_train_err,omp_train_err,las_train_err)))

        #PLOT 1c: Compare training error
        ax5 = plt.subplot(325)
        bins=np.histogram(np.hstack((model_train_err,omp_train_err,las_train_err)),bins=bin_num)[1]
        binwidth = bins[1]-bins[0]
        bins=np.concatenate((bins,[bins[-1]+binwidth]))
        n1, _ , _ = plt.hist(model_train_err, facecolor = mpl.colors.to_rgba(cols[0],.7), edgecolor=cols[0], label='GenMod', bins=bins)
        n2, _ , _ = plt.hist(omp_train_err, facecolor = mpl.colors.to_rgba(cols[1],.5), edgecolor=cols[1], label='OMP', bins=bins-binwidth/3)
        n3, _ , _ = plt.hist(las_train_err, facecolor = mpl.colors.to_rgba(cols[2],.3), edgecolor=cols[2],  label='Lasso', bins=bins-2*binwidth/3)
        if plotNoSparse:
            n4, _ , _ = plt.hist(model2_train_err, facecolor = mpl.colors.to_rgba(cols[3],.3), edgecolor=cols[3],  label='GenMod-NoSparse', bins=bins-2*binwidth/3)
            n3 = np.concatenate((n3,n4))
        max_height = np.max(np.concatenate((n1,n2,n3)))

        plt.xlim([0,bins[-1]*1.1])
        plt.xlabel('Relative training error')
        plt.ylabel('Frequency')
        plt.legend()

        #PLOT: Difference between training error
        ax6 = plt.subplot(326)
        ax6.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        n, _, _ = plt.hist(100*train_err_diff, bins=bin_num, facecolor = mpl.colors.to_rgba(cols[-1],.5), edgecolor=cols[-1])
        plt.xlabel(r'Percent Improvement')

    plt.savefig('/app/current_output/fig_presentation_data' + dataset + '_barplots.pdf')
    plt.show()

def plot_error_with_N(n_vec,fn,N_Sample_Sets,n_valid,u_data,y_data,dataset,c_ls=None,plot_coefs=True,plot_recons=True):

    N_tot = np.size(u_data)
    n_vec_len = np.size(n_vec)

    #Parameter determining how much to shift points to help with readability
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    dodge = 1.05

    #### Organize data
    if plot_coefs:
        mean_coef_err = np.zeros(n_vec_len)
        max_coef_err =  np.zeros(n_vec_len)
        min_coef_err =  np.zeros(n_vec_len)

        mean_coef_err_omp =  np.zeros(n_vec_len)
        max_coef_err_omp =  np.zeros(n_vec_len)
        min_coef_err_omp =  np.zeros(n_vec_len)

        mean_coef_err_las =  np.zeros(n_vec_len)
        max_coef_err_las =  np.zeros(n_vec_len)
        min_coef_err_las =  np.zeros(n_vec_len)

        mean_coef_err_noSparse = np.zeros(n_vec_len)
        max_coef_err_noSparse = np.zeros(n_vec_len)
        min_coef_err_noSparse = np.zeros(n_vec_len)

        mean_diff_coef_err = np.zeros(n_vec_len)
        min_diff_coef_err = np.zeros(n_vec_len)
        max_diff_coef_err = np.zeros(n_vec_len)

    if plot_recons:
        mean_recons_err = np.zeros(n_vec_len)
        max_recons_err = np.zeros(n_vec_len)
        min_recons_err = np.zeros(n_vec_len)

        mean_recons_err_omp = np.zeros(n_vec_len)
        max_recons_err_omp = np.zeros(n_vec_len)
        min_recons_err_omp = np.zeros(n_vec_len)

        mean_recons_err_las = np.zeros(n_vec_len)
        max_recons_err_las = np.zeros(n_vec_len)
        min_recons_err_las = np.zeros(n_vec_len)

        mean_diff_recons_err = np.zeros(n_vec_len)
        max_diff_recons_err = np.zeros(n_vec_len)
        min_diff_recons_err = np.zeros(n_vec_len)

        diff_recons_err = np.zeros(n_vec_len)
        diff_recons_err_min = np.zeros(n_vec_len)
        diff_recons_err_max = np.zeros(n_vec_len)

        mean_recons_err_noSparse = np.zeros(n_vec_len)
        max_recons_err_noSparse = np.zeros(n_vec_len)
        min_recons_err_noSparse = np.zeros(n_vec_len)


    for i in range(np.size(n_vec)):
        n = n_vec[i]

        #Store generative model + sparse error
        model_err = np.zeros(N_Sample_Sets)
        recons_err = np.zeros(N_Sample_Sets)

        #Store generative model without sparse vector error
        noSparse_err = np.zeros(N_Sample_Sets)
        recons_err_noSparse = np.zeros(N_Sample_Sets)

        #Store OMP error
        omp_err = np.zeros(N_Sample_Sets)
        recons_err_omp = np.zeros(N_Sample_Sets)

        #Store lasso error
        las_err = np.zeros(N_Sample_Sets)
        recons_err_las = np.zeros(N_Sample_Sets)


        for j in range(N_Sample_Sets):
            dmo = pickle.load(open(fn + '_n=' + str(n) + '_' + str(j) + '.pkl','rb'))
            omp = pickle.load(open(fn + '_n=' + str(n) + '_omp_' + str(j) + '.pkl','rb'))
            las = pickle.load(open(fn + '_n=' + str(n) + '_las_' + str(j) + '.pkl','rb'))

            #Validation data (data not used in optimization)
            data_indices = np.concatenate((dmo.sample_indices,dmo.valid_indices))
            not_data_indices = np.setdiff1d(range(N_tot),data_indices)
            valid_data_indices = random.sample(list(not_data_indices),n_valid)
            u_valid = u_data[valid_data_indices]
            u_valid_norm = np.linalg.norm(u_valid)

            c_ls_norm = np.linalg.norm(c_ls)

            y_valid = y_data[valid_data_indices]
            Psi = pcu.make_Psi(y_valid,dmo.mi_mat)

            #Only GenMod
            z = dmo.zdict['Iter:0'][-1,:]
            c_noSparse = dmo.zeta*dmu.decay_model(z,dmo.mi_c,dmo.beta_param_mat)

            #OMP
            c_omp = omp.coef_

            #LASSO
            c_las = las.coef_

            if plot_coefs:
                model_err[j] = np.linalg.norm(dmo.c-c_ls)/c_ls_norm
                omp_err[j] = np.linalg.norm(c_omp-c_ls)/c_ls_norm
                las_err[j] = np.linalg.norm(c_las-c_ls)/c_ls_norm
                noSparse_err[j] = np.linalg.norm(c_noSparse-c_ls)/c_ls_norm

            if plot_recons:
                recons_err[j] = np.linalg.norm(Psi @ dmo.c - u_valid)/u_valid_norm
                recons_err_omp[j] = np.linalg.norm(Psi @ c_omp - u_valid)/u_valid_norm
                recons_err_las[j] = np.linalg.norm(Psi @ c_las - u_valid)/u_valid_norm
                recons_err_noSparse[j] = np.linalg.norm(Psi @ c_noSparse - u_valid)/u_valid_norm

        if plot_coefs:
            mean_coef_err[i] = np.mean(model_err)
            max_coef_err[i] = np.max(model_err)
            min_coef_err[i] = np.min(model_err)

            mean_coef_err_omp[i] = np.mean(omp_err)
            max_coef_err_omp[i] = np.max(omp_err)
            min_coef_err_omp[i] = np.min(omp_err)

            mean_coef_err_las[i] = np.mean(las_err)
            max_coef_err_las[i] = np.max(las_err)
            min_coef_err_las[i] = np.min(las_err)

            mean_coef_err_noSparse[i] = np.mean(noSparse_err)
            max_coef_err_noSparse[i] = np.max(noSparse_err)
            min_coef_err_noSparse[i] = np.min(noSparse_err)

            mean_diff_coef_err[i] = np.mean((omp_err-model_err)/omp_err)
            min_diff_coef_err[i] = np.min((omp_err-model_err)/omp_err)
            max_diff_coef_err[i] = np.max((omp_err-model_err)/omp_err)

        if plot_recons:
            mean_recons_err[i] = np.mean(recons_err)
            max_recons_err[i] = np.max(recons_err)
            min_recons_err[i] = np.min(recons_err)

            mean_recons_err_omp[i] = np.mean(recons_err_omp)
            max_recons_err_omp[i] = np.max(recons_err_omp)
            min_recons_err_omp[i] = np.min(recons_err_omp)

            mean_recons_err_las[i] = np.mean(recons_err_las)
            max_recons_err_las[i] = np.max(recons_err_las)
            min_recons_err_las[i] = np.min(recons_err_las)

            mean_recons_err_noSparse[i] = np.mean(recons_err_noSparse)
            max_recons_err_noSparse[i] = np.max(recons_err_noSparse)
            min_recons_err_noSparse[i] = np.min(recons_err_noSparse)

            mean_diff_recons_err[i] = np.mean((recons_err_omp-recons_err)/recons_err_omp)
            min_diff_recons_err[i] = np.min((recons_err_omp-recons_err)/recons_err_omp)
            max_diff_recons_err[i] = np.max((recons_err_omp-recons_err)/recons_err_omp)

    #### Create figure
    fig = plt.figure(figsize=(6,7))

    if plot_recons:

        ax1 = plt.subplot(321)
        plt.errorbar(np.array(n_vec)/dodge,mean_recons_err,
            yerr = np.vstack((mean_recons_err-min_recons_err,max_recons_err-mean_recons_err)),
            marker = 'o', label = "GenMod")
        plt.errorbar(np.array(n_vec),mean_recons_err_omp,
            yerr = np.vstack((mean_recons_err_omp-min_recons_err_omp,max_recons_err_omp-mean_recons_err_omp)),
            marker = 's', label = "OMP")
        plt.errorbar(np.array(n_vec)*dodge,mean_recons_err_las,
            yerr = np.vstack((mean_recons_err_las-min_recons_err_las,max_recons_err_las-mean_recons_err_las)),
            marker = '^', label = "Lasso")
        plt.loglog()
        plt.ylim([np.min(np.concatenate([min_recons_err,min_recons_err_las,min_recons_err_omp]))*.5,
                np.max(np.concatenate([max_recons_err_las,max_recons_err,max_recons_err_omp]))*2])
        plt.ylabel(r'Rel. reconsturction error ($\varepsilon_u$)')
        plt.legend()

        ax3 = plt.subplot(323, sharex=ax1)
        plt.errorbar(np.array(n_vec)/dodge,mean_recons_err,
            yerr = np.vstack((mean_recons_err-min_recons_err,max_recons_err-mean_recons_err)),
            marker = 'o', label = "GenMod")
        plt.errorbar(np.array(n_vec),mean_recons_err_noSparse,
            yerr = np.vstack((mean_recons_err_noSparse-min_recons_err_noSparse,
                    max_recons_err_noSparse-mean_recons_err_noSparse)),
            marker = 's', color = cols[3], label = "GenMod-NoSparse")
        plt.loglog()
        plt.ylim([np.min(np.concatenate([min_recons_err,min_recons_err_noSparse]))*.5,
            np.max(np.concatenate([max_recons_err,max_recons_err_noSparse]))*2])
        plt.ylabel(r'Rel. reconstruction error ($\varepsilon_u$)')
        plt.legend()

        ax5 = plt.subplot(325, sharex=ax1)
        ax5.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        plt.errorbar(n_vec,100*mean_diff_recons_err,
            yerr = 100*np.vstack((mean_diff_recons_err-min_diff_recons_err,max_diff_recons_err-mean_diff_recons_err)),
            marker = 'o', color = cols[-1], label = "GenMod improvement \n on OMP error")

        plt.axhline(y=0,color='black',linestyle='--',linewidth=1)
        plt.xscale('log')
        plt.legend()
        plt.ylim([100*(np.min(min_diff_recons_err)-.1),100*(np.max(max_diff_recons_err)+.1)])
        plt.ylabel(r'Recons. error reduction ($\Delta \varepsilon_u$)')
        plt.xlabel(r'Number of samples ($N$)')

    if plot_coefs:
        ax2 = plt.subplot(322)
        plt.errorbar(np.array(n_vec)/dodge,mean_coef_err,
            yerr = np.vstack((mean_coef_err-min_coef_err,max_coef_err-mean_coef_err)),
            marker = 'o',
            label = "GenMod")
        plt.errorbar(np.array(n_vec),mean_coef_err_omp,
            yerr = np.vstack((mean_coef_err_omp-min_coef_err_omp,max_coef_err_omp-mean_coef_err_omp)),
            marker = 's',
            label = "OMP")
        plt.errorbar(np.array(n_vec)*dodge,mean_coef_err_las,
            yerr = np.vstack((mean_coef_err_las-min_coef_err_las,max_coef_err_las-mean_coef_err_las)),
            marker = '^',
            label = "Lasso")
        plt.loglog()
        plt.ylim([np.min(np.concatenate([min_coef_err,min_coef_err_las,min_coef_err_omp]))*.5,np.max(np.concatenate([max_coef_err_las,max_coef_err,max_coef_err_omp]))*2])
        plt.ylabel(r'Rel. coefficient error ($\varepsilon_c$)')
        plt.legend()

        ax4 = plt.subplot(324, sharex=ax2)
        plt.errorbar(np.array(n_vec)/dodge,mean_coef_err,
            yerr = np.vstack((mean_coef_err-min_coef_err,max_coef_err-mean_coef_err)),
            marker = 'o',
            label = "GenMod")
        plt.errorbar(np.array(n_vec),mean_coef_err_noSparse,
            yerr = np.vstack((mean_coef_err_noSparse-min_coef_err_noSparse,max_coef_err_noSparse-mean_coef_err_noSparse)),
            marker = 's',
            color = cols[3],
            label = "GenMod-NoSparse")
        plt.loglog()
        plt.ylim([np.min(np.concatenate([min_coef_err,min_coef_err_noSparse]))*.5,np.max(np.concatenate([max_coef_err,max_coef_err_noSparse]))*2])
        plt.ylabel(r'Rel. coefficient error ($\varepsilon_c$)')
        plt.legend()

        ax6 = plt.subplot(326, sharex=ax2)
        ax6.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        plt.errorbar(n_vec,100*mean_diff_coef_err,
            yerr = 100*np.vstack((mean_diff_coef_err-min_diff_coef_err,max_diff_coef_err-mean_diff_coef_err)),
            marker = 'o',
            color = cols[-1],
            label = "GenMod improvement \n on OMP error")
        plt.axhline(y=0,color='black',linestyle='--',linewidth=1)
        plt.xscale('log')
        plt.ylim([100*(np.min(min_diff_coef_err)-.1),100*(np.max(max_diff_coef_err)+.1)])
        plt.ylabel(r'Coef. error reduction ($\Delta \varepsilon_c$)')
        plt.xlabel(r'Number of samples ($N$)')
        plt.legend()

    plt.savefig('/app/current_output/fig_data' + dataset + '_increasing_N.pdf')
    plt.show()

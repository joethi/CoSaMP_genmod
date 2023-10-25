import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
import pickle
import random
import sys

import genmodpce.polynomial_chaos_utils as pcu
import genmodpce.decay_model_utils as dmu
import genmodpce.decay_models as decay_models

def plot_mean_error(N_sample_sets,fn,fn_c_ls,dataset,u_data,y_data,n_valid,bin_num=20):

    N_tot = np.size(u_data)
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

    c_ls = pickle.load(open(fn_c_ls + '.pkl','rb'))

    genmod_mean_error = np.zeros(N_sample_sets)
    lasso_mean_error = np.zeros(N_sample_sets)
    omp_mean_error = np.zeros(N_sample_sets)

    genmod_var_error = np.zeros(N_sample_sets)
    lasso_var_error = np.zeros(N_sample_sets)
    omp_var_error = np.zeros(N_sample_sets)

    model_recons_err = np.zeros(N_sample_sets)
    las_recons_err = np.zeros(N_sample_sets)
    omp_recons_err = np.zeros(N_sample_sets)

    genmod_decay_values = np.zeros((N_sample_sets,np.size(y_data,1)))

    for i in range(N_sample_sets):
        dmo = pickle.load(open(fn + '_' + str(i) + '.pkl','rb'))
        omp = pickle.load(open(fn + '_omp_' + str(i) + '.pkl','rb'))
        las = pickle.load(open(fn + '_las_' + str(i) + '.pkl','rb'))

        #OMP
        c_omp = omp.coef_

        #Lasso
        c_las = las.coef_

        genmod_mean_error[i] = np.absolute(dmo.c[0]-c_ls[0])/c_ls[0]
        omp_mean_error[i] = np.absolute(c_omp[0]-c_ls[0])/c_ls[0]
        lasso_mean_error[i] = np.absolute(c_las[0]-c_ls[0])/c_ls[0]

        genmod_var = np.sum(dmo.c[1:]**2)
        ls_var = np.sum(c_ls[1:]**2)
        omp_var = np.sum(c_omp[1:]**2)
        las_var = np.sum(c_las[1:]**2)
        genmod_var_error[i] = np.absolute(genmod_var-ls_var)/ls_var
        omp_var_error[i] = np.absolute(omp_var-ls_var)/ls_var
        lasso_var_error[i] = np.absolute(las_var-ls_var)/ls_var

        model_recons_err[i] = np.linalg.norm(dmo.c - c_ls)**2/np.linalg.norm(c_ls)**2
        omp_recons_err[i] = np.linalg.norm(c_omp - c_ls)**2/np.linalg.norm(c_ls)**2
        las_recons_err[i] = np.linalg.norm(c_las - c_ls)**2/np.linalg.norm(c_ls)**2

        genmod_decay_values[i] = dmo.z[1:1+np.size(y_data,1)]+np.log(2)*dmo.z[1+np.size(y_data,1):]

    mean_diff = omp_mean_error - genmod_mean_error
    var_diff = omp_var_error - genmod_var_error
    mse_diff = omp_recons_err - model_recons_err


    #Initialize figure
    fig = plt.figure(figsize=(5,7))

    recons_err_min = np.min(np.concatenate((model_recons_err,omp_recons_err,las_recons_err)))
    recons_err_max = np.max(np.concatenate((model_recons_err,omp_recons_err,las_recons_err)))

    #PLOT: Compare reconstruction error
    ax3 = plt.subplot(321)
    bins=np.histogram(np.hstack((model_recons_err,omp_recons_err,las_recons_err)),bins=bin_num)[1]
    binwidth = bins[1]-bins[0]
    bins=np.concatenate((bins,[bins[-1]+binwidth]))
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    n1, _ , _ = plt.hist(model_recons_err, facecolor = mpl.colors.to_rgba(cols[0],.7), edgecolor=cols[0], label='GenMod', bins=logbins)
    n2, _ , _ = plt.hist(omp_recons_err, facecolor = mpl.colors.to_rgba(cols[1],.5), edgecolor=cols[1], label='OMP', bins=logbins)
    n3, _ , _ = plt.hist(las_recons_err, facecolor = mpl.colors.to_rgba(cols[2],.3), edgecolor=cols[2],  label='Lasso', bins=logbins)

    plt.xlabel(r'Relative sum of squares error ')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.axis('tight')
    plt.legend()

    #bins = np.linspace(-.001,.009,11)
    plt.subplot(322)
    plt.hist(mse_diff, facecolor = mpl.colors.to_rgba(cols[-1],.7), edgecolor=cols[-1])
    plt.ylabel('Frequency')
    plt.xlabel('OMP - GenMod MSE')
    plt.axis('tight')

    mean_min = np.min(np.concatenate((genmod_mean_error,lasso_mean_error,omp_mean_error)))
    mean_max = np.max(np.concatenate((genmod_mean_error,lasso_mean_error,omp_mean_error)))

    #PLOT: Compare histograms of the mean value
    ax3 = plt.subplot(323)
    bins=np.histogram(np.hstack((genmod_mean_error,lasso_mean_error,omp_mean_error)),bins=bin_num)[1]
    binwidth = bins[1]-bins[0]
    bins=np.concatenate((bins,[bins[-1]+binwidth]))
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

    n1, _ , _ = plt.hist(genmod_mean_error, facecolor = mpl.colors.to_rgba(cols[0],.7), edgecolor=cols[0], label='GenMod', bins=logbins)
    n2, _ , _ = plt.hist(omp_mean_error, facecolor = mpl.colors.to_rgba(cols[1],.5), edgecolor=cols[1], label='OMP', bins=logbins)
    n3, _ , _ = plt.hist(lasso_mean_error, facecolor = mpl.colors.to_rgba(cols[2],.3), edgecolor=cols[2],  label='Lasso', bins=logbins)

#    plt.xlim([0,bins[-1]*1.1])
    plt.xlabel(r'Relative error of the mean')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.axis('tight')
    plt.legend()

    plt.subplot(324)
    #bins = np.linspace(-.02,.03,int((.03+.02)/.005)+1)
    plt.hist(mean_diff, facecolor = mpl.colors.to_rgba(cols[-1],.7), edgecolor=cols[-1])
    plt.ylabel('Frequency')
    plt.xlabel('OMP - GenMod Mean Error')

    plt.subplot(325)
    bins=np.histogram(np.hstack((genmod_var_error,lasso_var_error,omp_var_error)),bins=bin_num)[1]
    binwidth = bins[1]-bins[0]
    bins=np.concatenate((bins,[bins[-1]+binwidth]))
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

    n1, _ , _ = plt.hist(genmod_var_error, facecolor = mpl.colors.to_rgba(cols[0],.7), edgecolor=cols[0], label='GenMod', bins=logbins)
    n2, _ , _ = plt.hist(omp_var_error, facecolor = mpl.colors.to_rgba(cols[1],.5), edgecolor=cols[1], label='OMP', bins=logbins)
    n3, _ , _ = plt.hist(lasso_var_error, facecolor = mpl.colors.to_rgba(cols[2],.3), edgecolor=cols[2],  label='Lasso', bins=logbins)

#    plt.xlim([0,bins[-1]*1.1])
    plt.xlabel(r'Relative error of the variance')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.axis('tight')
    plt.legend()

    plt.subplot(326)
    #bins = np.linspace(-.2,1,13)
    plt.hist(var_diff, facecolor = mpl.colors.to_rgba(cols[-1],.7), edgecolor=cols[-1])
    plt.ylabel('Frequency')
    plt.xlabel('OMP - GenMod Variance Error')

    plt.savefig('/app/current_output/fig_' + dataset + '_barplots.pdf')
    plt.show()

    #Save statistics in text file
        #Save statistics in text file
    # lines = [['Method', 'Mean', 'Standard Deviation', 'Minimum', 'Maximum'],
    #     ['GenMod', np.mean(genmod_mean_error), np.std(genmod_mean_error), np.min(genmod_mean_error), np.max(genmod_mean_error)],
    #     ['OMP', np.mean(omp_mean_error), np.std(omp_mean_error), np.min(omp_mean_error), np.max(omp_mean_error)],
    #     ['Lasso', np.mean(lasso_mean_error), np.std(lasso_mean_error), np.min(lasso_mean_error), np.max(lasso_mean_error)]
    #     ]
    lines = [['GenMod'] + list(genmod_mean_error),['OMP'] + list(omp_mean_error),['Lasso'] + list(lasso_mean_error)]

    with open('/app/current_output/datafile_mean_' + dataset + '.txt','w') as f:
        for line in lines:
            f.write('\t'.join(map(str,line)))
            f.write('\n')

    # lines = [['Method', 'Mean', 'Standard Deviation', 'Minimum', 'Maximum'],
    #     ['GenMod', np.mean(genmod_var_error), np.std(genmod_var_error), np.min(genmod_var_error), np.max(genmod_var_error)],
    #     ['OMP', np.mean(omp_var_error), np.std(omp_var_error), np.min(omp_var_error), np.max(omp_var_error)],
    #     ['Lasso', np.mean(lasso_var_error), np.std(lasso_var_error), np.min(lasso_var_error), np.max(lasso_var_error)]
    #     ]
    lines = [
        ['GenMod'] + list(model_recons_err),
        ['OMP'] + list(omp_recons_err),
        ['Lasso'] + list(las_recons_err)
        ]
    with open('/app/current_output/datafile_mse_' + dataset + '.txt','w') as f:
        for line in lines:
            f.write('\t'.join(map(str,line)))
            f.write('\n')

    lines = [
        ['GenMod'] + list(genmod_var_error),
        ['OMP'] + list(omp_var_error),
        ['Lasso'] + list(lasso_var_error)
        ]
    with open('/app/current_output/datafile_variance_' + dataset + '.txt','w') as f:
        for line in lines:
            f.write('\t'.join(map(str,line)))
            f.write('\n')

    lines_header = np.array(['Mean','Standard Deviation','Minimum','Maximum'])
    lines_data = np.transpose(np.vstack((np.mean(genmod_decay_values,0),np.std(genmod_decay_values,0),np.min(genmod_decay_values,0),np.max(genmod_decay_values,0))))

    lines = np.vstack((lines_header,lines_data))
    with open('/app/current_output/datafile_exp_decay_rates_' + dataset + '.txt','w') as f:
        for line in lines:
            f.write('\t'.join(map(str,line)))
            f.write('\n')

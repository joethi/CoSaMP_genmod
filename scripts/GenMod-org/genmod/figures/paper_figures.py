import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
import pickle
import random
import sys
sys.path.append('/app/genmodpce/')
import decay_models
import numpy.linalg as la
import csv
import pandas as pd
import polynomial_chaos_utils as pcu
import decay_model_utils as dmu
import L_curve_utils_lasso as lcu
import colorsys
#import genmodpce.decay_models as decay_models

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def plot_coef_comparison(coefs_fitted,c_omp,c_las,initial_sign,c_ls,dataset_name):
    """
    Plot 1: Comparison of coefficients obtained with GenMod, OMP and Lasso
        with reference soltuion.
    Plot 2: Shows whether the sign of each coefficient obtained with GenMod
        is ocrrect and if it was flipped by the sparse vector.
    """

    xrange = np.linspace(1,np.size(c_ls),np.size(c_ls))

    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure(figsize=(5.5,6))

    #Plot 1
    ax1 = plt.subplot(211)
    plt.grid(True)
    plt.scatter(xrange,np.absolute(c_ls),facecolor='gray',label="Least Squares",marker='s')
    plt.scatter(xrange,np.absolute(coefs_fitted),label="GenMod",marker='o',alpha=.7)
    plt.scatter(xrange,np.absolute(c_omp),label="OMP",marker='s',alpha=.7)
    plt.scatter(xrange,np.absolute(c_las),label="IRW-Lasso",marker='^',alpha=.7)
    plt.loglog()
    plt.xlim([.9,np.max(xrange)])
    plt.ylim([10**-9,np.max(np.concatenate([c_ls,coefs_fitted,c_omp,c_las]))*4])
    plt.ylabel('Coefficient magnitude')
    plt.legend()


    #Plot 2
    label_vec = (np.sign(coefs_fitted)*np.sign(c_ls)).astype(int)
    ci = label_vec==1 #indicies of coefficients with correct sign
    ii = label_vec==-1 #indices of coefficients with incorrect sign
    max_incorrect = np.max(c_ls[ii]) #Number of incorrectly labeled coefficients
    omp_vec = (np.sign(coefs_fitted)*initial_sign).astype(int)
    fi = omp_vec==-1 #coefficients whose sign flipped

    plt.subplot(212, sharex=ax1)
    plt.grid(True)
    plt.scatter(xrange,np.absolute(c_ls),facecolor='gray',label="Least Squares",marker='s',s=3)
    plt.scatter(xrange[ci],np.absolute(coefs_fitted[ci]),
        facecolor='none',edgecolor=cols[0])
    plt.scatter(xrange[ci],np.absolute(coefs_fitted[ci]),
        facecolor=mpl.colors.to_rgba(cols[0],.6),edgecolor=cols[0],
        label="GenMod: Correct Sign") #For some reason this command only adds outline to legend markers.

    plt.scatter(xrange[ii],np.absolute(coefs_fitted[ii]),
        facecolor='none',edgecolor=cols[7])
    plt.scatter(xrange[ii],np.absolute(coefs_fitted[ii]),
        facecolor=mpl.colors.to_rgba(cols[7],.6),edgecolor=cols[7],
        label="GenMod: Incorrect Sign") #For some reason this command only adds outline to legend markers.

    plt.scatter(xrange[fi & ci],np.absolute(coefs_fitted[fi & ci]),marker='*',
        edgecolor='black',facecolor=cols[0],
        s=35,label='Sign flipped (correctly)')
    plt.scatter(xrange[fi & ii],np.absolute(coefs_fitted[fi & ii]),marker='*',
        edgecolor='black',facecolor=cols[7],
            s=35,label='Sign flipped (incorrectly)')

    plt.loglog()
    plt.xlim([.9,np.max(xrange)])
    plt.ylim([10**-9,np.max(np.concatenate([c_ls,coefs_fitted]))*4])
    plt.ylabel('Coefficient magnitude')
    plt.xlabel('Index of PC coefficient')
    plt.legend(ncol=2)

    plt.savefig('/app/current_output/fig_' + dataset_name + '_coef_mag.pdf')
    plt.show()

    return(np.sum(fi & ci),np.sum(fi & ii),max_incorrect)

def plot_coef_and_recons(N_sample_sets,fn,dataset,u_data,y_data,indices,n_valid,bin_num=20,plotTrain=False,plotNoSparse=False,plotRecons=True,plotCoefs=True,c_ls = None):

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
        #model_recons_err_2 = np.zeros(N_sample_sets)
        las_recons_err = np.zeros(N_sample_sets)
        omp_recons_err = np.zeros(N_sample_sets)
        recons_err_diff = np.zeros(N_sample_sets)
        if plotNoSparse:
            model2_recons_err = np.zeros(N_sample_sets)

    for i in range(N_sample_sets):
        dmo = pickle.load(open(fn + '/GenMod/' + dataset + '_' + str(i) + '.pkl','rb'))
        #dmo_2 = pickle.load(open(fn + '_LCurve_' + str(i) + '.pkl','rb'))

        # omp = pickle.load(open(fn + '_omp_' + str(i) + '.pkl','rb'))
        # las = pickle.load(open(fn + '_wlas3_' + str(i) + '.pkl','rb'))

        #Validation data (data not used in optimization)
        # data_indices = np.concatenate((dmo.sample_indices,dmo.valid_indices))
        # not_data_indices = np.setdiff1d(range(N_tot),data_indices)

        data_indices = indices.loc[i].to_numpy()
        not_data_indices = np.setdiff1d(range(np.size(u_data)),data_indices)
        valid_data_indices = random.sample(list(not_data_indices),n_valid)
        u_valid = u_data[valid_data_indices]
        y_valid = y_data[valid_data_indices]
        Psi = pcu.make_Psi(y_valid,dmo.mi_mat)


        u_train = np.concatenate((dmo.u_data,dmo.u_valid))
        Psi_train = np.vstack((dmo.Psi,dmo.Psi_valid))

        #OMP

        las = pd.read_csv(fn + '/IRW-Lasso/' + dataset + '_las_' + str(i)+'.csv')
        c_las = las.iloc[:,0].to_numpy()

        omp = pd.read_csv(fn + '/OMP/' + dataset + '_omp_' + str(i)+'.csv')
        c_omp = omp.iloc[:,0].to_numpy()

        # c_omp = omp.coef_
        #
        # #Lasso
        # c_las = las.coef_

        if plotCoefs:
            model_coef_err[i] = np.linalg.norm(dmo.c-c_ls)/np.linalg.norm(c_ls)
            omp_coef_err[i] = np.linalg.norm(c_omp-c_ls)/np.linalg.norm(c_ls)
            las_coef_err[i] = np.linalg.norm(c_las-c_ls)/np.linalg.norm(c_ls)
            coef_err_diff[i] = (las_coef_err[i] - model_coef_err[i])/las_coef_err[i]

        if plotTrain:
            model_train_err[i] = np.linalg.norm(Psi_train @ dmo.c - u_train)/np.linalg.norm(u_train)
            omp_train_err[i] = np.linalg.norm(Psi_train @ c_omp - u_train)/np.linalg.norm(u_train)
            las_train_err[i] = np.linalg.norm(Psi_train @ c_las - u_train)/np.linalg.norm(u_train)
            train_err_diff[i] = (las_train_err[i] - model_train_err[i])/las_train_err[i]

            if plotNoSparse:
                z = dmo.zdict['Iter:0'][-1,:]
                c = dmo.zeta*dmu.decay_model(z,dmo.mi_c,dmo.beta_param_mat)
                model2_train_err[i] = np.linalg.norm(Psi_train @ c - u_train)/np.linalg.norm(u_train)

        if plotRecons:
            model_recons_err[i] = np.linalg.norm(Psi @ dmo.c - u_valid)/np.linalg.norm(u_valid)
            #model_recons_err_2[i] = np.linalg.norm(Psi @ dmo_2.c - u_valid)/np.linalg.norm(u_valid)
            omp_recons_err[i] = np.linalg.norm(Psi @ c_omp - u_valid)/np.linalg.norm(u_valid)
            las_recons_err[i] = np.linalg.norm(Psi @ c_las - u_valid)/np.linalg.norm(u_valid)
            recons_err_diff[i] = (las_recons_err[i] - model_recons_err[i])/las_recons_err[i]
            print('i='+str(i))
            print(las_recons_err[i])
            if plotNoSparse:
                z = dmo.zdict['Iter:0'][-1,:]
                c = dmo.zeta*dmu.decay_model(z,dmo.mi_c,dmo.beta_param_mat)
                model2_recons_err[i] = np.linalg.norm(Psi @ c - u_valid)/np.linalg.norm(u_valid)
        if i==0:
            plt.figure()
            plt.hist(Psi @ dmo.c - u_valid)
            plt.xlabel('GenMod err')

            plt.figure()
            plt.hist(Psi @ c_omp - u_valid)
            plt.xlabel('OMP err')

    #Initialize figure
    fig = plt.figure(figsize=(6,7))

    if plotCoefs:
        coef_err_min = np.min(np.concatenate((model_coef_err,omp_coef_err,las_coef_err)))
        coef_err_max = np.max(np.concatenate((model_coef_err,omp_coef_err,las_coef_err)))

        #PLOT: Compare coefficient error
        ax1 = plt.subplot(321)
        bins=np.histogram(np.hstack((model_coef_err,omp_coef_err,las_coef_err)),bins=bin_num)[1]
        # binwidth = bins[1]-bins[0]
        # n1, _ , _ = plt.hist(model_coef_err, facecolor = mpl.colors.to_rgba(cols[0],.7), edgecolor=cols[0], label='GenMod', bins=bins)
        # n2, _ , _ = plt.hist(omp_coef_err, facecolor = mpl.colors.to_rgba(cols[1],.5), edgecolor=cols[1], label='OMP', bins=bins+binwidth/3)
        # n3, _ , _ = plt.hist(las_coef_err, facecolor = mpl.colors.to_rgba(cols[2],.3), edgecolor=cols[2],  label='IRW-Lasso', bins=bins+2*binwidth/3)

        n, bins, patches = plt.hist(
            [np.sort(model_coef_err), np.sort(omp_coef_err), np.sort(las_coef_err)],
            bins=bins,
            histtype='bar',
            linewidth=0,
            label=[
                'GenMod',
                'OMP',
                'IRW-Lasso'
            ]
        )

        plt.xlim([0,bins[-1]*1.1])
        plt.xlabel(r'Relative coefficient error ($\varepsilon_c$)')
        plt.ylabel('Frequency')
        #plt.xscale('log')
        plt.legend()

        #PLOT: Difference between coefficient error
        ax2 = plt.subplot(322)
        ax2.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        n, _, _ = plt.hist(100*coef_err_diff,bins=bin_num, facecolor = mpl.colors.to_rgba(cols[-1],.5), edgecolor=cols[-1])
        plt.xlabel(r'Percent Improvement ($\Delta \varepsilon_c$)')

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
        #logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        #binwidth = logbins[1]-logbins[0]
        #_, _ , _ = plt.hist(omp_recons_err, facecolor = 'none', edgecolor=cols[-1],  hatch='\\', bins=bins - binwidth/3)
        #_, _ , _ = plt.hist(las_recons_err, facecolor = 'none', edgecolor=cols[-1],  hatch='//', lw=.5, bins=bins - 2*binwidth/3)



        # n1, _ , _ = plt.hist(model_recons_err, facecolor = mpl.colors.to_rgba(cols[0],.7), edgecolor=lighten_color(cols[0],1.5), label='GenMod', bins=bins)
        # n2, _ , _ = plt.hist(omp_recons_err, facecolor = mpl.colors.to_rgba(cols[1],.5), edgecolor=lighten_color(cols[1],1.5), label='OMP', bins=bins - binwidth/3)
        # n3, _ , _ = plt.hist(las_recons_err, facecolor = mpl.colors.to_rgba(cols[2],.3), edgecolor=lighten_color(cols[2],1.5), label='IRW-Lasso',bins=bins - 2*binwidth/3)
        # n1, _ , _ = plt.hist(model_recons_err, facecolor = 'none', edgecolor=lighten_color(cols[0],1.2), bins=bins)
        # n2, _ , _ = plt.hist(omp_recons_err, facecolor = 'none', edgecolor=lighten_color(cols[1],1.2), bins=bins - binwidth/3)
        # n3, _ , _ = plt.hist(las_recons_err, facecolor = 'none', edgecolor=lighten_color(cols[2],1.2), bins=bins - 2*binwidth/3)

        n, bins, patches = plt.hist(
            [np.sort(model_recons_err), np.sort(omp_recons_err), np.sort(las_recons_err)],
            bins=bins,
            histtype='bar',
            linewidth=0,
            label=[
                'GenMod',
                'OMP',
                'IRW-Lasso'
            ]
        )
        print(n)
        if plotNoSparse:
            n4, _ , _ = plt.hist(model2_recons_err, facecolor = mpl.colors.to_rgba(cols[3],.3), edgecolor=cols[3],  label='GenMod-NoSparse', bins=bins - 2*binwidth/3)
            n3 = np.concatenate((n3,n4))
        #max_height = np.max(np.concatenate((n1,n2,n3)))

        plt.xlim([0,bins[-1]*1.1])
        plt.ylim([0,np.max(n)+5])        
        #ax3.set_xscale('log')
        plt.xlabel(r'Relative reconstruction error ($\varepsilon_u$)')
        plt.ylabel('Frequency')
        plt.legend()

        #PLOT: Difference between reconstruction error
        ax4 = plt.subplot(324)
        ax4.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        n, _, _ = plt.hist(100*recons_err_diff, bins=bin_num, facecolor = mpl.colors.to_rgba(cols[-1],.5), edgecolor=cols[-1])
        plt.xlabel(r'Percent Improvement ($\Delta \varepsilon_u$)')

        print(str.format('Number of Delta u < 0: {:g}',np.size(np.where(recons_err_diff<0))))

        #Print varaince and mean results
        print('GenMod Reconstruction mean: ' + str(np.mean(model_recons_err)))
        print('OMP Reconstruction mean: ' + str(np.mean(omp_recons_err)))
        print('IRw-Lasso Reconstruction mean: ' + str(np.mean(las_recons_err)))

        print('GenMod Reconstruction variance: ' + str(np.var(model_recons_err)))
        print('OMP Reconstruction variance: ' + str(np.var(omp_recons_err)))
        print('IRw-Lasso Reconstruction variance: ' + str(np.var(las_recons_err)))

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
        n3, _ , _ = plt.hist(las_train_err, facecolor = mpl.colors.to_rgba(cols[2],.3), edgecolor=cols[2],  label='IRW-Lasso', bins=bins-2*binwidth/3)
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
        plt.xlabel(r'Percent Improvement ($\Delta \varepsilon_{u,train}$)')

    plt.savefig('/app/current_output/fig_' + dataset + '_barplots.pdf')
    plt.show()

def plot_error_with_N(n_vec,fn,N_Sample_Sets,n_valid,u_data,y_data,dataset,indices,c_ls=None,plot_coefs=True,plot_recons=True):

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

        mean_coef_err2 = np.zeros(3)
        max_coef_err2 =  np.zeros(3)
        min_coef_err2 =  np.zeros(3)

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

        mean_diff_coef_err2 = np.zeros(n_vec_len)
        min_diff_coef_err2 = np.zeros(n_vec_len)
        max_diff_coef_err2 = np.zeros(n_vec_len)

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

        mean_diff_recons_err2 = np.zeros(n_vec_len)
        max_diff_recons_err2 = np.zeros(n_vec_len)
        min_diff_recons_err2 = np.zeros(n_vec_len)

        diff_recons_err = np.zeros(n_vec_len)
        diff_recons_err_min = np.zeros(n_vec_len)
        diff_recons_err_max = np.zeros(n_vec_len)

        diff_recons_err2 = np.zeros(n_vec_len)
        diff_recons_err_min2 = np.zeros(n_vec_len)
        diff_recons_err_max2 = np.zeros(n_vec_len)

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
            print(j)
            dmo = pickle.load(open(fn + '/n=' + str(n) + '/GenMod/' + dataset + '_n=' + str(n) + '_' + str(j) + '.pkl','rb'))

            las = pd.read_csv(fn + '/n=' + str(n) + '/IRW-Lasso/' + dataset + '_n=' + str(n) + '_las_' + str(j)+'.csv')
            c_las = las.iloc[:,0].to_numpy()

            omp = pd.read_csv(fn + '/n=' + str(n) + '/OMP/' + dataset + '_n=' + str(n) + '_omp_' + str(j)+'.csv')
            c_omp = omp.iloc[:,0].to_numpy()

            #Validation data (data not used in optimization)
            data_indices = indices.loc[i].to_numpy()
            not_data_indices = np.setdiff1d(range(N_tot),data_indices)
            valid_data_indices = random.sample(list(not_data_indices),n_valid)
            u_valid = u_data[valid_data_indices]
            u_valid_norm = np.linalg.norm(u_valid)
            y_valid = y_data[valid_data_indices]
            Psi = pcu.make_Psi(y_valid,dmo.mi_mat)





            #Only GenMod
            z = dmo.zdict['Iter:0'][-1,:]
            c_noSparse = dmo.zeta*dmu.decay_model(z,dmo.mi_c,dmo.beta_param_mat)


            c_ls_norm = np.linalg.norm(c_ls)
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

            mean_diff_coef_err[i] = np.mean((las_err-model_err)/las_err)
            min_diff_coef_err[i] = np.min((las_err-model_err)/las_err)
            max_diff_coef_err[i] = np.max((las_err-model_err)/las_err)

            mean_diff_coef_err2[i] = np.mean((omp_err-model_err)/omp_err)
            min_diff_coef_err2[i] = np.min((omp_err-model_err)/omp_err)
            max_diff_coef_err2[i] = np.max((omp_err-model_err)/omp_err)

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

            mean_diff_recons_err[i] = np.mean((recons_err_las-recons_err)/recons_err_las)
            min_diff_recons_err[i] = np.min((recons_err_las-recons_err)/recons_err_las)
            max_diff_recons_err[i] = np.max((recons_err_las-recons_err)/recons_err_las)

            mean_diff_recons_err2[i] = np.mean((recons_err_omp-recons_err)/recons_err_omp)
            min_diff_recons_err2[i] = np.min((recons_err_omp-recons_err)/recons_err_omp)
            max_diff_recons_err2[i] = np.max((recons_err_omp-recons_err)/recons_err_omp)

        if n==40:
            print('Reconstruction Error Diff OMP for n=40:')
            print((recons_err_omp-recons_err)/recons_err_omp)
            print('Coefficient Error Diff OMP for n=40:')
            print((omp_err-model_err)/omp_err)
            print('Reconstruction Error Diff IRW-Lasso for n=40:')
            print((recons_err_las-recons_err)/recons_err_las)
            print('Coefficient Error Diff IRW-Lasso for n=40:')
            print((las_err-model_err)/las_err)
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
            marker = '^', label = "IRW-Lasso")
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
        plt.errorbar(n_vec,100*mean_diff_recons_err2,
            yerr = 100*np.vstack((mean_diff_recons_err2-min_diff_recons_err2,max_diff_recons_err2-mean_diff_recons_err2)),
            marker = 's', color = cols[1], label = "GenMod improvement \n on OMP error")
        plt.errorbar(n_vec,100*mean_diff_recons_err,
            yerr = 100*np.vstack((mean_diff_recons_err-min_diff_recons_err,max_diff_recons_err-mean_diff_recons_err)),
            marker = '^', color = cols[2], label = "GenMod improvement \n on IWR-Lasso error")

        plt.axhline(y=0,color='black',linestyle='--',linewidth=1)
        plt.xscale('log')
        plt.legend()
        plt.ylim([100*(np.min(min_diff_recons_err)-.6),100*(np.max(max_diff_recons_err)+.6)])
        plt.ylabel(r'Recons. error reduction ($\Delta \varepsilon_u$)')
        plt.xlabel(r'Number of samples ($N$)')

    if plot_coefs:
        ax2 = plt.subplot(322)
        plt.errorbar(np.array(n_vec)/dodge,mean_coef_err,
            yerr = np.vstack((mean_coef_err-min_coef_err,max_coef_err-mean_coef_err)),
            marker = 'o',
            label = "GenMod")
        # plt.errorbar(np.array([29,40,80])/(dodge**2),mean_coef_err2,
        #     yerr = np.vstack((mean_coef_err2-min_coef_err2,max_coef_err2-mean_coef_err2)),
        #     marker = 'o',
        #     label = "GenMod-LCurve")
        plt.errorbar(np.array(n_vec),mean_coef_err_omp,
            yerr = np.vstack((mean_coef_err_omp-min_coef_err_omp,max_coef_err_omp-mean_coef_err_omp)),
            marker = 's',
            label = "OMP")
        plt.errorbar(np.array(n_vec)*dodge,mean_coef_err_las,
            yerr = np.vstack((mean_coef_err_las-min_coef_err_las,max_coef_err_las-mean_coef_err_las)),
            marker = '^',
            label = "IRW-Lasso")
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
        plt.errorbar(n_vec,100*mean_diff_coef_err2,
            yerr = 100*np.vstack((mean_diff_coef_err2-min_diff_coef_err2,max_diff_coef_err2-mean_diff_coef_err2)),
            marker = 's',
            color = cols[1],
            label = "GenMod improvement \n on OMP error")
        plt.errorbar(n_vec,100*mean_diff_coef_err,
            yerr = 100*np.vstack((mean_diff_coef_err-min_diff_coef_err,max_diff_coef_err-mean_diff_coef_err)),
            marker = '^',
            color = cols[2],
            label = "GenMod improvement \n on IRW-Lasso error")
        plt.axhline(y=0,color='black',linestyle='--',linewidth=1)
        plt.xscale('log')
        plt.ylim([100*(np.min(min_diff_coef_err)-.6),100*(np.max(max_diff_coef_err)+.6)])
        plt.ylabel(r'Coef. error reduction ($\Delta \varepsilon_c$)')
        plt.xlabel(r'Number of samples ($N$)')
        plt.legend()

    plt.savefig('/app/current_output/fig_' + dataset + '_increasing_N.pdf')
    plt.show()

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

    for i in range(N_sample_sets):
        dmo = pickle.load(open(fn + '_' + str(i) + '.pkl','rb'))
        omp = pickle.load(open(fn + '_omp_' + str(i) + '.pkl','rb'))
        las = pickle.load(open(fn + '_wlas_' + str(i) + '.pkl','rb'))

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

    mean_diff = omp_mean_error - genmod_mean_error
    var_diff = omp_var_error - genmod_var_error
    mse_diff = omp_recons_err - model_recons_err


    #Initialize figure
    fig = plt.figure(figsize=(8,5))

    recons_err_min = np.min(np.concatenate((model_recons_err,omp_recons_err,las_recons_err)))
    recons_err_max = np.max(np.concatenate((model_recons_err,omp_recons_err,las_recons_err)))

    #PLOT: Compare reconstruction error
    ax3 = plt.subplot(231)
    bins=np.histogram(np.hstack((model_recons_err,omp_recons_err,las_recons_err)),bins=bin_num)[1]
    binwidth = bins[1]-bins[0]
    bins=np.concatenate((bins,[bins[-1]+binwidth]))
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    n1, _ , _ = plt.hist(model_recons_err, facecolor = mpl.colors.to_rgba(cols[0],.7), edgecolor=cols[0], label='GenMod', bins=logbins)
    n2, _ , _ = plt.hist(omp_recons_err, facecolor = mpl.colors.to_rgba(cols[1],.5), edgecolor=cols[1], label='OMP', bins=logbins)
    n3, _ , _ = plt.hist(las_recons_err, facecolor = mpl.colors.to_rgba(cols[2],.3), edgecolor=cols[2],  label='IRW-Lasso', bins=logbins)

    plt.xlabel(r'Relative sum of squares error ')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.axis('tight')
    plt.legend()

    bins = np.linspace(-.001,.009,11)
    plt.subplot(234)
    plt.hist(mse_diff, facecolor = mpl.colors.to_rgba(cols[-1],.7), edgecolor=cols[-1],bins=bins)
    plt.ylabel('Frequency')
    plt.xlabel('OMP - GenMod MSE')
    plt.axis('tight')

    mean_min = np.min(np.concatenate((genmod_mean_error,lasso_mean_error,omp_mean_error)))
    mean_max = np.max(np.concatenate((genmod_mean_error,lasso_mean_error,omp_mean_error)))

    #PLOT: Compare histograms of the mean value
    ax3 = plt.subplot(232)
    bins=np.histogram(np.hstack((genmod_mean_error,lasso_mean_error,omp_mean_error)),bins=bin_num)[1]
    binwidth = bins[1]-bins[0]
    bins=np.concatenate((bins,[bins[-1]+binwidth]))
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

    n1, _ , _ = plt.hist(genmod_mean_error, facecolor = mpl.colors.to_rgba(cols[0],.7), edgecolor=cols[0], label='GenMod', bins=logbins)
    n2, _ , _ = plt.hist(omp_mean_error, facecolor = mpl.colors.to_rgba(cols[1],.5), edgecolor=cols[1], label='OMP', bins=logbins)
    n3, _ , _ = plt.hist(lasso_mean_error, facecolor = mpl.colors.to_rgba(cols[2],.3), edgecolor=cols[2],  label='IRW-Lasso', bins=logbins)

#    plt.xlim([0,bins[-1]*1.1])
    plt.xlabel(r'Relative error of the mean')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.axis('tight')
    plt.legend()

    plt.subplot(235)
    bins = np.linspace(-.02,.03,int((.03+.02)/.005)+1)
    plt.hist(mean_diff, facecolor = mpl.colors.to_rgba(cols[-1],.7), edgecolor=cols[-1],bins=bins)
    plt.ylabel('Frequency')
    plt.xlabel('OMP - GenMod Mean Error')

    plt.subplot(233)
    bins=np.histogram(np.hstack((genmod_var_error,lasso_var_error,omp_var_error)),bins=bin_num)[1]
    binwidth = bins[1]-bins[0]
    bins=np.concatenate((bins,[bins[-1]+binwidth]))
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

    n1, _ , _ = plt.hist(genmod_var_error, facecolor = mpl.colors.to_rgba(cols[0],.7), edgecolor=cols[0], label='GenMod', bins=logbins)
    n2, _ , _ = plt.hist(omp_var_error, facecolor = mpl.colors.to_rgba(cols[1],.5), edgecolor=cols[1], label='OMP', bins=logbins)
    n3, _ , _ = plt.hist(lasso_var_error, facecolor = mpl.colors.to_rgba(cols[2],.3), edgecolor=cols[2],  label='IRW-Lasso', bins=logbins)

#    plt.xlim([0,bins[-1]*1.1])
    plt.xlabel(r'Relative error of the variance')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.axis('tight')
    plt.legend()

    plt.subplot(236)
    bins = np.linspace(-.2,1,13)
    plt.hist(var_diff, facecolor = mpl.colors.to_rgba(cols[-1],.7), edgecolor=cols[-1],bins=bins)
    plt.ylabel('Frequency')
    plt.xlabel('OMP - GenMod Variance Error')

    plt.savefig('/app/current_output/fig_' + dataset + '_barplots.pdf')
    plt.show()

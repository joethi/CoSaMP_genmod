import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
out_dir = '/home/jothi/CoSaMP_genNN/output/titan_ppr/results/csaug13/d=21/p=3/ompcv/N=7k_test'
err = []; S_lst = []; trn_err = []
for t_ind in range(62):
    df_err = pd.read_csv(f'{out_dir}/err_test_S_test{t_ind+7}_tst_ind{t_ind}.csv')
    err.append(df_err['vlerr_test'].to_numpy())
    trn_err.append(df_err['trnerr_test'].to_numpy())
    S_lst.append(t_ind+7)
plt.figure(1)
plt.plot(S_lst,err,'r*--',label='valid')
plt.plot(S_lst,trn_err,'bo--',label='train')
plt.xlabel('S')
plt.ylabel('Relative validation error')
plt.legend()
plt.grid()
plt.savefig(f'{out_dir}/S_vs_err.png',dpi=300)



import seaborn as sns
#import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
#/home/jothi/CoSaMP_genNN/output 
############# Main Section ############
# loading dataset using seaborn
out_dir_ini = "../output/test"
df = sns.load_dataset('tips')
# pairplot with hue sex
sns.pairplot(df,hue='day')
# to show
#plt.show()
plt.savefig(f"{out_dir_ini}/pair_plot.png")
import pdb; pdb.set_trace()
# This code is contributed by Deepanshu Rustagi.

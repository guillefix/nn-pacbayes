import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib
import glob
# d["train_acc"]

prefix="new_shifted_init_sweep"
dataset="mnist"
net="cnn"
number_layers=4

prefix="tanh_centered_fc_shifted_init_sweep"
prefix="uncentered_fc_shifted_init_sweep"
prefix="fc_shifted_init_sweep"
net="fc"
number_layers=1
dataset="mnist"
dataset="EMNIST"

#%%
d = pd.read_csv("results/"+prefix+"_1_"+dataset+"_nn_training_results.txt", sep="\t", comment="#")
d = d.sort_values("shifted_init_shift")
x=d["shifted_init_shift"]
test_sensitivities=d["test_sensitivity"]
test_accuracies=d["test_acc"]


# shift=0.5
# shifts=[-5.0,-4.0,-3.0,-2.0,-1.0,0.0,0.5,1.0,2.0,3.0,4.0,5.0]
# shifts=[-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0]
# shifts=[0.0]
shifts=x
hist_datas=[]
mean_n1s=[]
for i,shift in enumerate(shifts):
    # print(shift)
    d = pd.DataFrame()
    for filename in glob.glob("results/index_funs_probs_*_"+prefix+"_*_"+str(shift)+"_"+dataset+"_"+net+"_"+str(number_layers)+"*.txt"):
        # print(filename)
        d = pd.concat([d,pd.read_csv(filename, header=None, names=["index","fun","ent","n1s"],delim_whitespace=True, \
            dtype={"index":np.int, "fun":"str","ent":np.float,"n1s":np.int})],axis=0)

    hist_data = plt.hist(d["n1s"], bins=len(d["n1s"].unique()), label=shift)
    mean_n1s.append(d["n1s"].mean())
    hist_datas.append(hist_data)

plt.legend()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(x, test_sensitivities, '-', c="red", label = 'Sensitivity')
# lns2 = ax.plot(x, np.array(mean_n1s)/100, '-', label = 'T')
ax2 = ax.twinx()
ax3 = ax.twinx()
lns3 = ax2.plot(shifts, np.array(mean_n1s), '-k', label = '$\langle T\\rangle$')
lns2 = ax3.plot(x, test_accuracies, '-b', label = 'Accuracy')

from matplotlib.ticker import FormatStrFormatter
ax3.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax3.yaxis.set_major_locator(plt.MaxNLocator(6))
# ax3.xaxis.set_major_locator(plt.MaxNLocator(7))
ax3.spines['left'].set_position(('outward', 60))
ax3.spines["left"].set_visible(True)
ax3.yaxis.set_label_position('left')
ax3.yaxis.set_ticks_position('left')
plt.subplots_adjust(left=0.23, right=0.9, top=0.9, bottom=0.15)
# added these three lines
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc="upper left")
ax.grid()
ax.set_xlabel("shift")
# ax.set_ylabel("Sensitivity")
ax.set_ylabel("Test sensitivity")
ax2.set_ylabel("$\langle T\\rangle$")
ax3.set_ylabel("Test accuracy")

# plt.savefig("sensitivity_T_vs_shift_100mnist_cnn_sigmab0.png")
# plt.savefig("accuracy_sensitivity_T_vs_shift_"+dataset+"_"+net+"_"+str(number_layers)+"_"+prefix+".png")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib

# d["train_acc"]

d = pd.read_csv("results/new_shifted_init_sweep_1_mnist_nn_training_results.txt", sep="\t", comment="#")
d = d.sort_values("shifted_init_shift")
d.columns
x=d["shifted_init_shift"]
y=d["test_sensitivity"]
y=d["test_acc"]

# plt.plot(x,y)
# plt.xlabel("shift")
# plt.ylabel("Sensitivity")
# plt.savefig("sensitivity_vs_b_cnn_100mnist_sigmab0.png")

# shift=0.5
shifts=[-5.0,-4.0,-3.0,-2.0,-1.0,0.0,0.5,1.0,2.0,3.0,4.0,5.0]
hist_datas=[]
mean_n1s=[]
for i,shift in enumerate(shifts):
    d = pd.read_csv("results/index_funs_probs_0_new_shifted_init_sweep_"+"{0:g}".format(shift)+"_1_mnist__"+str(shift)+"_mnist_cnn_4_none_0000.txt", header=None, names=["index","fun","ent","n1s"],delim_whitespace=True, \
        dtype={"index":np.int, "fun":"str","ent":np.float,"n1s":np.int})

    hist_data = plt.hist(d["n1s"], bins=len(d["n1s"].unique()))
    mean_n1s.append(d["n1s"].mean())
    hist_datas.append(hist_data)

# for i,hist_data in enumerate(hist_datas[::-1]):
#     plt.bar(hist_data[1][:-1],hist_data[0]/np.sum(hist_data[0]),width=1, log=False, alpha=0.8,color=(i/12,i/12,i/12), label=shifts[::-1][i])
#
# plt.legend()
# plt.xlabel("T")
# plt.ylabel("Probability")

# d.plot.hist("n1s")
# plt.plot(shifts,np.array(mean_n1s)/100)
fig = plt.figure()
ax = fig.add_subplot(111)

# lns1 = ax.plot(x, y, '-', label = 'Sensitivity')
lns1 = ax.plot(x, y, '-', label = 'Accuracy')
# lns2 = ax.plot(shifts, np.array(mean_n1s)/100, '-', label = 'T')
ax2 = ax.twinx()
lns3 = ax2.plot(shifts, np.array(mean_n1s), '-r', label = '$\langle T\\rangle$')

# added these three lines
lns = lns1+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
ax.grid()
ax.set_xlabel("shift")
# ax.set_ylabel("Sensitivity")
ax.set_ylabel("Accuracy")
ax2.set_ylabel("$\langle T\\rangle$")

# plt.savefig("sensitivity_T_vs_shift_100mnist_cnn_sigmab0.png")
plt.savefig("accuracy_T_vs_shift_100mnist_cnn_sigmab0.png")

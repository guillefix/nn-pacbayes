import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results_folder = "results/"

prefix = "GP_vs_prob_"
prefix = "GPMC_vs_prob_"
bounds = pd.read_csv(results_folder+prefix+"bounds.txt",comment="#", header='infer',sep="\t")

d1 = pd.read_csv("GPMC_logPs.txt",comment="#", header=None,sep="\t")
logPs = d1[1]
funs = d1[0]

# bounds["prob"]
bounds["logP"]


# d = pd.read_csv("../unique_prob_set_1e7_7_40_40_1_relu.txt", delim_whitespace=True, header=None)
d = pd.read_csv("../unique_prob_set_1e7_2_7_40_40_1_1.000000_0.000000_relu_fun_samples.txt", delim_whitespace=True, header=None)
d[0]

GPMC_logPs = []
empirical_logPs = []
nonnan_funs = []
for i,fun in enumerate(funs):
    if not np.isnan(logPs[i]):
        empirical_logPs.append(np.log10(d[d[0]==fun][1]/1e7).iloc[0])
        GPMC_logPs.append(logPs[i]/np.log(10))
        nonnan_funs.append(fun)

GPMC_logPs = np.array(GPMC_logPs)
empirical_logPs = np.array(empirical_logPs)

np.argmin(GPMC_logPs - empirical_logPs)

# np.isnan(GPMC_logPs[147])

nonnan_funs.index("11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100011110")


"00000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
'00000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'

nonnan_funs[112]

GPMC_logPs[145]
empirical_logPs[145]

for i in range(len(GPMC_logPs)):
    empirical_logPs[i] - GPMC_logPs[i]

d = d[slice(2,len(d),3)][:-19]

import matplotlib.pyplot as plt
%matplotlib
# plt.scatter(np.log10(d[2]/1e7),bounds["prob"]/np.log(10))
# plt.scatter(np.log10(d[1]/1e7),bounds["logP"]/np.log(10))
plt.scatter(empirical_logPs,GPMC_logPs)
plt.xlabel("Empirical log(P)")
plt.ylabel("GP EP log(P)")
plt.savefig("empirical_vs_GPEP_sigmaw50.0_sigmab0.0_logP_1e7_7_40_40_1_relu.png")

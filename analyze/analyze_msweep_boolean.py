import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib

d = pd.read_csv("results/all_funs_nn_random_labels_boolean_msweep__0.0_boolean_fc_2_avg_00.txt", header=None, delim_whitespace=True, names=["index", "fun", "ent", "t"])
training_results = pd.read_csv("results/nn_random_labels_boolean_msweep_nn_training_results.txt", header='infer',sep="\t", comment="#")
# plt.plot(training_results["m"], training_results["test_error"])
training_results["test_acc_std"]
training_results.columns
num_ms = len(training_results["m"])
plt.errorbar(training_results["m"], training_results["test_error"],training_results["test_acc_std"])
plt.xscale("log")
plt.yscale("log")
plt.xlabel("m")
plt.ylabel("Expected error")
plt.savefig("expected_error_vs_m_boolean_n7_fc2.png")

d.dtypes
d["fun"] = d["fun"].map(lambda x: str(x))
d["m"] = d["fun"].map(lambda x: len(x))

# str(d["fun"].head(10).iloc[0])

#%%
import matplotlib.pyplot as plt

from scipy.stats import entropy

ms = d["m"].unique()
mmax = max(ms)
tot_cntss = []
entropies = []
len(ms)
for m in ms:
    print(m)
    m=ms[16]
    fun_cnts = d[d["m"] == m].groupby("fun", as_index=False).count().sort_values("index", ascending=False)
    tot_cnts = sum(fun_cnts["index"])
    freqs = list(1.0*fun_cnts["index"]/tot_cnts)
    tot_cntss.append(tot_cnts)
    entropies.append(entropy(freqs))

    plt.plot(freqs, label=m, color=str(m/mmax))
    plt.yscale("log")
    plt.xscale("log")
    # plt.legend()
    #
    # plt.xlabel("Rank")
    # plt.ylabel("Probability")
    # plt.savefig("prob_rank_msweep_boolean_fc_2_1e6.png")


ms[15]
entropies
tot_cntss[12]

# all the total counts should have been 1e6. Something went wrong.

tot_cntss
plt.scatter(ms, entropies)
plt.scatter(ms, tot_cntss)

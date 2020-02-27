import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#%matplotlib

# train_set="01100010000011100000110101000000101100001010000101001000000110010100000000010000000000000001000000101001100000010010000101000000"
# target_fun="00110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011"
train_set2 = "01001100010100101100001000100010000010010011000010000010001010010000000001100000010000100000000000100010010000011000100011000000"

train_sets = [train_set2]*6 + ["01100010000011100000110101000000101100001010000101001000000110010100000000010000000000000001000000101001100000010010000101000000"]
#train_sets = [train_set2]*7

funs = ["10001000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000","11111111111111111111111111111111010111110111111101111111011111111111111111111111111111111111111111111111111111111111111111111111","11111111111111111111111111111111011101110111011101110111011101111111111111111111111111111111111111111111111111111111111111111111","11111111111111111111111111111111010101010101010101010101010101011111111111111111111111111111111101010101010101010101010101010101","00110011001100110000000000000000001100110011001100000000000000000000000000000000000000000000000000000000000000000000000000000000","00001111000011110000111100001111000011110000111100001111000011111111111111111111111111111111111111111111111111111111111111111111","00110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011"]

def compute_gen_error(testfun,target_testfun):
    assert len(testfun) == len(target_testfun)
    return len([x for i,x in enumerate(testfun) if x==target_testfun[i]])/len(testfun)

#%%

# i=5
# fun=funs[i]

# overtraining_epochs = 64
overtraining_epochs = 0

for i,fun in list(enumerate(funs))[6:]:
    # if i<=5:
    #     continue
    train_set = train_sets[i]
    train_set_indices = [i for i in range(len(train_set)) if train_set[i]=="1"]
    test_set_indices = [i for i in range(len(train_set)) if train_set[i]=="0"]
    target_fun = fun
    # abi_sample = pd.read_csv("results/sgd_vs_bayes/"+str(i)+"_abi_sample_net_2hl_sorted.txt", sep="\t", header=None,names=["freq","fun","testerror"])
    abi_sample = pd.read_csv("results/sgd_vs_bayes/"+str(i)+"_abi_sample_net_2hl_sorted_testfun.csv")
    sgd_sample = pd.read_csv("results/sgd_vs_bayes/"+str(i)+"_sgd_vs_bayes_32_40_2_8_sgd_ce_"+str(overtraining_epochs)+"__nn_train_functions_sorted.txt", header=None, delim_whitespace=True, names=["freq","testfun"])

    target_testfun = "".join([x for i,x in enumerate(target_fun) if i in test_set_indices])

    # abi_sample
    sgd_tot_samples = sum(sgd_sample["freq"])
    abi_tot_samples = sum(abi_sample["freq"])
    print(str(i),"total number of samples (sgd/abi)", sgd_tot_samples,abi_tot_samples)

    print(str(i),"number of functions (sgd/abi)", len(sgd_sample),len(abi_sample))

    # abi_sample["testfun"] = abi_sample["fun"].apply(lambda fun: "".join([x for i,x in enumerate(fun) if i in test_set_indices]))
    # abi_sample = abi_sample.groupby("testfun").sum()
    sgd_sample = sgd_sample.set_index("testfun")
    abi_sample = abi_sample.set_index("testfun")

    # abi_sample.to_csv("results/sgd_vs_bayes/"+str(i)+"_abi_sample_net_2hl_sorted_testfun.csv")
    gen_error_sgd_sample = list(map(lambda x: compute_gen_error(x, target_testfun), sgd_sample.index))
    gen_error_abi_sample = list(map(lambda x: compute_gen_error(x, target_testfun), abi_sample.index))
    plt.hist(gen_error_sgd_sample, alpha=0.5, density=True, bins=np.linspace(0.7,1.0,20), label="SGD sampling");
    plt.hist(gen_error_abi_sample, alpha=0.5, density=True, bins=np.linspace(0.7,1.0,20), label="ABI sampling");
    plt.legend()
    plt.xlabel("Test error")
    plt.savefig(str(i)+"_test_error_histograms_sgd_fc_32_"+str(overtraining_epochs)+"__8_sgd_ce_vs_abi_1_7_2x40_1.png")
    plt.close()
    ##ABI vs SGD

    abi_freqs = []
    sgd_freqs = []
    test_funs = []

    for test_fun,row in sgd_sample.iterrows():
        sgd_freq = row["freq"]
        if test_fun in abi_sample.index:
            if sgd_freq > 3:
                abi_freq = abi_sample.loc[test_fun]["freq"]
                if abi_freq > 3:
                    abi_freqs.append(abi_freq)
                    sgd_freqs.append(sgd_freq)
                    test_funs.append(test_fun)

    normalized_abi_freqs = np.array(abi_freqs)/abi_tot_samples
    normalized_sgd_freqs = np.array(sgd_freqs)/sgd_tot_samples
    # #%%
    # plt.scatter(normalized_abi_freqs, normalized_sgd_freqs)
    # # plt.scatter(normalized_sgd_freqs2, normalized_sgd_freqs)
    # # plt.scatter(logPUs_GP, normalized_sgd_freqs)
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.xlim([normalized_abi_freqs.min()*0.5,normalized_abi_freqs.max()*1.5])
    # plt.ylim([normalized_sgd_freqs.min()*0.5,normalized_sgd_freqs.max()*1.5])
    # plt.xlabel("ABI probabilities")
    # plt.ylabel("SGD probabilities")
    # # plt.xlabel("SGD probabilities (Early stopping)")
    # # plt.ylabel("SGD probabilities (Overtrain)")
    # plt.plot([normalized_sgd_freqs.min()*0.5,normalized_abi_freqs.max()*2],[normalized_sgd_freqs.min()*0.5,normalized_abi_freqs.max()*2], c='k')
    # #%%
    #     # plt.savefig("sgd_fc_32___8_sgd_ce_wait1_vs_sgd_wait64_7_2x40_1_above5.png")
    # #%%

    # H, xedges, yedges = np.histogram2d(normalized_abi_freqs, normalized_sgd_freqs)
    # h,xedges,yedges,_ = plt.hist2d(np.log(normalized_abi_freqs), np.log(normalized_sgd_freqs), weights=normalized_abi_freqs, bins=30)
    # h,xedges,yedges,_ = plt.hist2d(np.log10(normalized_abi_freqs), np.log10(normalized_sgd_freqs), weights=normalized_sgd_freqs, bins=30)
    h,xedges,yedges,_ = plt.hist2d(np.log10(normalized_abi_freqs), np.log10(normalized_sgd_freqs), bins=30)
    plt.close()
    # h = h/np.maximum(1e-6,h.max(axis=1, keepdims=True))
    h = h/np.maximum(1e-6,h.max(axis=0, keepdims=True))
    plt.imshow(np.rot90(h))

    from math import floor,ceil
    rx=(len(xedges)-1)/(xedges[-1]-xedges[0])
    orders_of_mag = range(int(ceil(min(xedges))-0.5/rx),int(floor(max(xedges))+0.5/rx))
    tick_places = list(map(lambda x: rx*(x-xedges[0]), orders_of_mag))
    # tick_places = range(3,30,5)
    plt.xticks(tick_places,["$10^{{{0:.0f}}}$".format(orders_of_mag[ii]) for ii,t in enumerate(tick_places)]);
    ry=(len(xedges)-1)/(yedges[-1]-yedges[0])
    orders_of_mag = range(int(ceil(min(yedges))-0.5/ry),int(floor(max(yedges))+0.5/ry))
    tick_places = list(map(lambda x: 29.5-ry*(x-yedges[0]), orders_of_mag))
    # tick_places = list(map(lambda x: 30-np.argmin(np.abs(yedges+x))+0.5, range(0,int(abs(min(yedges)))+1)))
    plt.yticks(tick_places,["$10^{{{0:.0f}}}$".format(orders_of_mag[ii]) for ii,t in enumerate(tick_places)]);
    plt.colorbar()
    plt.xlabel("ABI probabilities")
    plt.ylabel("SGD probabilities")

    if len(normalized_abi_freqs)>1:
        x_min = normalized_abi_freqs.min()
        x_max = normalized_abi_freqs.max()
        y_min = normalized_sgd_freqs.min()
        y_max = normalized_sgd_freqs.max()
        plt.plot([(np.log10(np.maximum(y_min-x_min,0)+x_min)-xedges[0])*rx-0.5,(np.log10(np.minimum(y_max-x_max,0)+x_max)-xedges[0])*rx-0.5],[29.5-(np.log10(np.maximum(x_min-y_min,0)+y_min)-yedges[0])*ry,29.5-(np.log10(np.minimum(x_max-y_max,0)+y_max)-yedges[0])*ry], c='g')
    # plt.plot([0,0],[30,30],c='k')
    #%%

    # # plt.savefig("sgd_fc_32___8_sgd_ce_wait64_vs_abi_7_2x40_1_above3_sgd_weighted_row_normalized.png")
    plt.savefig(str(i)+"_sgd_vs_bayes2_"+str(overtraining_epochs)+".png")
    plt.close()
    # plt.hist2d(np.log(normalized_abi_freqs), np.log(normalized_sgd_freqs))
    # from matplotlib.colors import LogNorm
    # plt.hist2d(np.log(normalized_abi_freqs), np.log(normalized_sgd_freqs), bins=40, norm=LogNorm())

# [x for ii,x in enumerate(yedges) if ii in tick_places]
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
# train_sets = [train_set2]*7

funs = ["10001000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000","11111111111111111111111111111111010111110111111101111111011111111111111111111111111111111111111111111111111111111111111111111111","11111111111111111111111111111111011101110111011101110111011101111111111111111111111111111111111111111111111111111111111111111111","11111111111111111111111111111111010101010101010101010101010101011111111111111111111111111111111101010101010101010101010101010101","00110011001100110000000000000000001100110011001100000000000000000000000000000000000000000000000000000000000000000000000000000000","00001111000011110000111100001111000011110000111100001111000011111111111111111111111111111111111111111111111111111111111111111111","00110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011"]

# funs = ["10001000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000","11111111111111111111111111111111010111110111111101111111011111111111111111111111111111111111111111111111111111111111111111111111","11111111111111111111111111111111011101110111011101110111011101111111111111111111111111111111111111111111111111111111111111111111","11111111111111111111111111111111010101010101010101010101010101011111111111111111111111111111111101010101010101010101010101010101","00110011001100110000000000000000001100110011001100000000000000000000000000000000000000000000000000000000000000000000000000000000","00001111000011110000111100001111000011110000111100001111000011111111111111111111111111111111111111111111111111111111111111111111"]

def compute_gen_error(testfun,target_testfun):
    assert len(testfun) == len(target_testfun)
    return len([x for i,x in enumerate(testfun) if x==target_testfun[i]])/len(testfun)

#%%

# i=5
# fun=funs[i]

# overtraining_epochs = 64
overtraining_epochs = 0

batch_size = 1
learning_rate="_0.01"
learning_rate=""

# first_name="ABI"
first_name="GP_EP"
second_name="SGD"

# %matplotlib
# first_sample["freq"].unique()
# second_sample["freq"].unique()

for i,fun in list(enumerate(funs)):
    # if i<=5:
    #     continue
    #%%
    train_set = train_sets[i]
    train_set_indices = [i for i in range(len(train_set)) if train_set[i]=="1"]
    test_set_indices = [i for i in range(len(train_set)) if train_set[i]=="0"]
    target_fun = fun
    # abi_sample = pd.read_csv("results/sgd_vs_bayes/"+str(i)+"_abi_sample_net_2hl_sorted.txt", sep="\t", header=None,names=["freq","fun","testerror"])
    for ii,name in enumerate([first_name,second_name]):
        if name == "SGD":
            thing = pd.read_csv("results/sgd_vs_bayes/"+str(i)+"_sgd_vs_bayes_32_40_2_"+str(batch_size)+"_sgd_ce_"+str(overtraining_epochs)+learning_rate+"__nn_train_functions_sorted.txt", header=None, delim_whitespace=True, names=["freq","testfun"])
        elif name == "ABI":
            thing = pd.read_csv("results/sgd_vs_bayes/"+str(i)+"_abi_sample_net_2hl_sorted_testfun.csv")
        elif name == "GP_EP":
            thing = pd.read_csv("results/sgd_vs_bayes/"+str(i)+"_sgd_vs_bayes_gpep_32_40_2_8_sgd_ce_"+str(overtraining_epochs)+"__nn_train_functions_sorted.txt", header=None, delim_whitespace=True, names=["freq","testfun"])
        else:
            raise NotImplementedError

        if ii == 0: first_sample = thing
        elif ii == 1: second_sample = thing

    target_testfun = "".join([x for i,x in enumerate(target_fun) if i in test_set_indices])

    # abi_sample
    first_tot_samples = sum(first_sample["freq"])
    second_tot_samples = sum(second_sample["freq"])
    print(str(i),"total number of samples ("+first_name+"/"+second_name+")", first_tot_samples,second_tot_samples)

    print(str(i),"number of functions ("+first_name+"/"+second_name+")", len(first_sample),len(second_sample))

    # abi_sample["testfun"] = abi_sample["fun"].apply(lambda fun: "".join([x for i,x in enumerate(fun) if i in test_set_indices]))
    # abi_sample = abi_sample.groupby("testfun").sum()
    first_sample = first_sample.set_index("testfun")
    second_sample = second_sample.set_index("testfun")

    #%%
    '''ERROR HISTOGRAMS'''
    # abi_sample.to_csv("results/sgd_vs_bayes/"+str(i)+"_abi_sample_net_2hl_sorted_testfun.csv")
    gen_error_first_sample = list(map(lambda x: compute_gen_error(x, target_testfun), first_sample.index))
    first_sample["testerror"] = np.array(1) - gen_error_first_sample
    gen_error_second_sample = list(map(lambda x: compute_gen_error(x, target_testfun), second_sample.index))
    second_sample["testerror"] = np.array(1) - gen_error_second_sample
    plt.hist(gen_error_second_sample, alpha=0.5, weights=second_sample["freq"], density=True, bins=np.linspace(0.7,1.0,20), label=second_name+" sampling");
    plt.hist(gen_error_first_sample, alpha=0.5, weights=first_sample["freq"], density=True, bins=np.linspace(0.7,1.0,20), label=first_name+" sampling");
    plt.legend()
    plt.xlabel("Test error")
    # plt.savefig(str(i)+"_test_error_histograms_sgd_fc_32_"+str(overtraining_epochs)+"__8_sgd_ce_vs_abi_1_7_2x40_1.png")
    # plt.savefig(str(i)+"_test_error_histograms_sgd_fc_32_"+str(overtraining_epochs)+"__8_sgd_ce_vs_gpep_1_7_2x40_1.png")
    plt.savefig(str(i)+"_"+first_name+"_vs_"+second_name+"_test_error_histograms_fc_32_"+str(overtraining_epochs)+"_"+learning_rate+"__"+str(batch_size)+"_1_7_2x40_1.png")
    plt.close()
    #%%
    # continue

    '''RANK PLOTS'''
    sorted_second_freqs = np.sort(second_sample["freq"])[::-1]
    sorted_first_freqs = np.sort(first_sample["freq"])[::-1]
    plt.plot(range(1,len(sorted_second_freqs)+1),sorted_second_freqs/second_tot_samples, label=second_name+" sampling")
    plt.plot(range(1,len(sorted_first_freqs)+1),sorted_first_freqs/first_tot_samples, label=first_name+" sampling")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Rank")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(str(i)+"_"+first_name+"_vs_"+second_name+"_rank_plots_fc_32_"+str(overtraining_epochs)+"_"+learning_rate+"__"+str(batch_size)+"_1_7_2x40_1.png")
    plt.close()
    #%%

    '''PROBABILITY-PROBABILITY SCATTER HISTOGRAMS'''
    second_freqs = []
    first_freqs = []
    test_funs = []
    errors = []

    # first_sample.sort_values("testerror",ascending=True, inplace=True)
    # second_sample.sort_values("testerror",ascending=True, inplace=True)

    for test_fun,row in first_sample.iterrows():
        first_freq = row["freq"]
        error = row["testerror"]*96
        if test_fun in second_sample.index:
            if first_freq > 0:
                second_freq = second_sample.loc[test_fun]["freq"]
                if second_freq > 0:
                    second_freqs.append(second_freq)
                    first_freqs.append(first_freq)
                    test_funs.append(test_fun)
                    errors.append(error)
        else:
            second_freqs.append(1)
            first_freqs.append(first_freq)
            test_funs.append(test_fun)
            errors.append(error)

    for test_fun,row in second_sample.iterrows():
        second_freq = row["freq"]
        error = row["testerror"]*96
        if test_fun not in first_sample.index:
            second_freqs.append(second_freq)
            first_freqs.append(1)
            test_funs.append(test_fun)
            errors.append(error)

    normalized_second_freqs = np.array(second_freqs)/second_tot_samples
    normalized_first_freqs = np.array(first_freqs)/first_tot_samples

    #%%
    # %matplotlib
    plt.scatter(errors,normalized_second_freqs, label=second_name+" sampling",alpha=0.5)
    # plt.scatter(range(1,len(normalized_second_freqs)+1),normalized_second_freqs, label=second_name+" sampling")
    # plt.plot(range(1,len(normalized_first_freqs)+1),normalized_first_freqs, label=first_name+" sampling")
    plt.scatter(errors,normalized_first_freqs, label=first_name+" sampling",alpha=0.5)
    # plt.scatter(range(1,len(normalized_first_freqs)+1),normalized_first_freqs, label=first_name+" sampling")
    plt.yscale("log")
    # plt.xscale("log")
    plt.xlim([0.9*np.min(errors),1.1*np.max(errors)])
    plt.ylim([0.9*1e-6,1.1*1e0])
    plt.xlabel("Test error")
    # plt.xlabel("Rank")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(str(i)+"_"+first_name+"_vs_"+second_name+"_error_scatter_plot_fc_32_"+str(overtraining_epochs)+"_"+learning_rate+"__"+str(batch_size)+"_1_7_2x40_1.png")
    # plt.close()
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

    # np.sort(normalized_first_freqs)[::-1]
    # np.sort(normalized_second_freqs)[::-1]

    # H, xedges, yedges = np.histogram2d(normalized_abi_freqs, normalized_sgd_freqs)
    # h,xedges,yedges,_ = plt.hist2d(np.log(normalized_abi_freqs), np.log(normalized_sgd_freqs), weights=normalized_abi_freqs, bins=30)
    # h,xedges,yedges,_ = plt.hist2d(np.log10(normalized_second_freqs), np.log10(normalized_first_freqs), weights=normalized_second_freqs, bins=30)

    ###################################
    # h,xedges,yedges,_ = plt.hist2d(np.log10(normalized_second_freqs), np.log10(normalized_first_freqs), bins=30)
    # plt.close()
    # # h = h/np.maximum(1e-6,h.max(axis=1, keepdims=True))
    # # h = h/np.maximum(1e-6,h.max(axis=0, keepdims=True))
    # # plt.imshow(np.rot90(h))
    # Z = h*(10**xedges[1:]) + 1e-6
    # Z = h + 1
    #
    # plt.pcolor(yedges,xedges, Z.T,
    #            # norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
    #            # norm=colors.PowerNorm(gamma=0.3),
    #            cmap='PuBu_r')
    #
    # from math import floor,ceil
    # rx=(len(xedges)-1)/(xedges[-1]-xedges[0])
    # orders_of_mag = range(int(ceil(min(xedges))-0.5/rx),int(floor(max(xedges))+0.5/rx))
    # tick_places = list(map(lambda x: rx*(x-xedges[0]), orders_of_mag))
    # # tick_places = range(3,30,5)
    # plt.xticks(tick_places,["$10^{{{0:.0f}}}$".format(orders_of_mag[ii]) for ii,t in enumerate(tick_places)]);
    # ry=(len(xedges)-1)/(yedges[-1]-yedges[0])
    # orders_of_mag = range(int(ceil(min(yedges))-0.5/ry),int(floor(max(yedges))+0.5/ry))
    # tick_places = list(map(lambda x: 29.5-ry*(x-yedges[0]), orders_of_mag))
    # # tick_places = list(map(lambda x: 30-np.argmin(np.abs(yedges+x))+0.5, range(0,int(abs(min(yedges)))+1)))
    # plt.yticks(tick_places,["$10^{{{0:.0f}}}$".format(orders_of_mag[ii]) for ii,t in enumerate(tick_places)]);
    # plt.colorbar()
    # plt.xlabel(second_name+" probabilities")
    # plt.ylabel(first_name+" probabilities")
    #
    # if len(normalized_second_freqs)>1:
    #     x_min = normalized_second_freqs.min()
    #     x_max = normalized_second_freqs.max()
    #     y_min = normalized_first_freqs.min()
    #     y_max = normalized_first_freqs.max()
    #     plt.plot([(np.log10(np.maximum(y_min-x_min,0)+x_min)-xedges[0])*rx-0.5,(np.log10(np.minimum(y_max-x_max,0)+x_max)-xedges[0])*rx-0.5],[29.5-(np.log10(np.maximum(x_min-y_min,0)+y_min)-yedges[0])*ry,29.5-(np.log10(np.minimum(x_max-y_max,0)+y_max)-yedges[0])*ry], c='g')
    # # plt.plot([0,0],[30,30],c='k')
    # #%%
    #
    # import matplotlib.colors as colors
    #
    #%%
    cmap = plt.cm.viridis  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    import matplotlib as mpl
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0, 10, 11)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    %matplotlib
    fig = plt.figure(num=None, figsize=(7,7), dpi=80, facecolor='w', edgecolor='k')
    x_min = normalized_second_freqs.min()
    x_max = normalized_second_freqs.max()
    y_min = normalized_first_freqs.min()
    y_max = normalized_first_freqs.max()
    min_val = np.minimum(y_min,x_min)
    max_val = np.maximum(y_max,x_max)
    plt.scatter(normalized_second_freqs,normalized_first_freqs,c=errors,cmap=cmap, norm=norm)
    plt.yscale("log")
    plt.xscale("log")
    xs = [0.5*np.minimum(min_val,1e-6),np.minimum(1.5,1.5*max_val)]
    ys = [0.5*np.minimum(min_val,1e-6),np.minimum(1.5,1.5*max_val)]
    plt.xlim(xs)
    plt.ylim(ys)
    plt.plot(xs,ys,c=np.array([10, 166, 10,125])/255)
    plt.grid(color="0.7")
    plt.xlabel(second_name+" sampling probability", fontsize=14)
    plt.ylabel(first_name+" sampling probability", fontsize=14)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    ax2 = fig.add_axes([0.91, 0.1, 0.03, 0.8])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
        spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')

    # plt.gca().yaxis.get_major_ticks()[-1].label.set_text("10+")

    plt.gca().set_yticklabels(list(range(0,10))+["10+"])
    # ax2.yaxis.get_major_ticks()[-1].label.set_fontsize(14)
    plt.savefig(str(i)+"_scatter_"+first_name+"_vs_"+second_name+"_"+str(overtraining_epochs)+"_"+learning_rate+"_"+str(batch_size)+".png")
    # plt.ylim([y_min*0.9,y_max*1.1])
    # plt.xlim([x_min*0.9,x_max*1.1])
    # # # plt.savefig("sgd_fc_32___8_sgd_ce_wait64_vs_abi_7_2x40_1_above3_sgd_weighted_row_normalized.png")
    # # plt.savefig(str(i)+"_sgd_vs_bayes_gpep_"+str(overtraining_epochs)+".png")
    # plt.savefig(str(i)+"_"+first_name+"_vs_"+second_name+"_"+str(overtraining_epochs)+".png")
    plt.close()
    #%%
    #####################################

    # plt.hist2d(np.log(normalized_abi_freqs), np.log(normalized_sgd_freqs))
    # from matplotlib.colors import LogNorm
    # plt.hist2d(np.log(normalized_abi_freqs), np.log(normalized_sgd_freqs), bins=40, norm=LogNorm())

# [x for ii,x in enumerate(yedges) if ii in tick_places]

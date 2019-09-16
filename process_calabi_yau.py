import numpy as np

exec("inputs="+open("/home/guillefix/Downloads/ConfigurationMatrices.txt","r").read())

def process(x):
    flat_x = np.array(x).flatten()
    return np.concatenate([flat_x,np.zeros(180-len(flat_x))])

inputs = [process(x) for x in inputs]

inputs = np.array(inputs)

inputs
exec("targets="+open("/home/guillefix/Downloads/SymmetryDataCoarse.txt","r").read())
targets=np.array(targets)
targets.shape

# import matplotlib.pyplot as plt
np.sum(targets)/len(targets)
# plt.plot(targets)

np.savez("datasets/calabiyau.npz",inputs=inputs,targets=targets)

# inputs.std(0)
#
# inputs.mean(0)

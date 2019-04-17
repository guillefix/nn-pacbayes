from math import log
import numpy as np
def KC_LZ(string):
    n=len(string)
    s = '0'+string
    c=1
    l=1
    i=0
    k=1
    k_max=1
    stop=0

    while stop==0:
        if s[i+k] != s[l+k]:
            if k>k_max:
                k_max=k # k_max stores the length of the longest pattern in the LA that has been matched somewhere in the SB

            i=i+1 # we increase i while the bit doesn't match, looking for a previous occurence of a pattern. s[i+k] is scanning the "search buffer" (SB)

            if i==l: # we stop looking when i catches up with the first bit of the "look-ahead" (LA) part.
                c=c+1 # If we were actually compressing, we would add the new token here. here we just count recounstruction STEPs
                l=l+k_max # we move the beginning of the LA to the end of the newly matched pattern.

                if l+1>n: # if the new LA is beyond the ending of the string, then we stop.
                    stop=1

                else: #after STEP,
                    i=0 # we reset the searching index to beginning of SB (beginning of string)
                    k=1 # we reset pattern matching index. Note that we are actually matching against the first bit of the string, because we added an extra 0 above, so i+k is the first bit of the string.
                    k_max=1 # and we reset max lenght of matched pattern to k.
            else:
                k=1 #we've finished matching a pattern in the SB, and we reset the matched pattern length counter.

        else: # I increase k as long as the pattern matches, i.e. as long as s[l+k] bit string can be reconstructed by s[i+k] bit string. Note that the matched pattern can "run over" l because the pattern starts copying itself (see LZ 76 paper). This is just what happens when you apply the cloning tool on photoshop to a region where you've already cloned...
            k=k+1

            if l+k>n: # if we reach the end of the string while matching, we need to add that to the tokens, and stop.
                c=c+1
                stop=1



    # a la Lempel and Ziv (IEEE trans inf theory it-22, 75 (1976),
    # h(n)=c(n)/b(n) where c(n) is the kolmogorov complexity
    # and h(n) is a normalised measure of complexity.
    complexity=c;

    #b=n*1.0/np.log2(n)
    #complexity=c/b;

    return complexity


def calc_KC(s):
    L = len(s)
    if s == '0'*L or s == '1'*L:
        return np.log2(L)
    else:
        return np.log2(L)*(KC_LZ(s)+KC_LZ(s[::-1]))/2.0


def log2(x):
    return log(x)/log(2.0)

def entropy(f):
    n0=0
    n=len(f)
    for char in f:
        if char=='0':
            n0+=1
    n1=n-n0
    if n1 > 0 and n0 > 0:
        return log2(n) - (1.0/n)*(n0*log2(n0)+n1*log2(n1))
    else:
        return 0

# inputs = [[int(l) for l in "{0:07b}".format(i)] for i in range(0,2**input_dim)]
# inputs_str = ["{0:07b}".format(i) for i in range(0,2**input_dim)]
# inp_dict = {"{0:07b}".format(i):i for i in range(0,2**input_dim)}

def neigh(x,h):
    n=[]
    if h==1:
        for i in range(len(x)):
            y=x[:]
            y[i]=(x[i]+1)%2
            n.append(y)
        return n
    if h==2:
        for i in range(len(x)):
            for j in range(i+1,len(x)):
                y=x[:]
                y[i]=(x[i]+1)%2
                y[j]=(x[j]+1)%2
                n.append(y)
        return n

# fun=funs[10000]

def hamming_comp(inputs_str,f, h):
    e=0
    # inp_dict = {"{0:07b}".format(i):i for i in range(0,2**input_dim)}
    # inp_dict
    inp_dict = {x:i for i,x in enumerate(inputs_str)}
    for i,inp in enumerate(inputs_str):
        for n in neigh([int(l) for l in inp],h):
            n_str = "".join([str(x) for x in n])
            if f[inp_dict[inp]] != f[inp_dict[n_str]]:
                e+=1
    return e/(len(inputs_str)*len(neigh([int(l) for l in inputs_str[0]],h)))

def hamming_comp_cum(inputs_str,f,hh):
    tot=0
    for h in range(1,hh+1):
        tot += hamming_comp(inputs_str,f, h)
    return tot

def crit_sample_ratio(inputs_str,f):
    e=0
    inp_dict = {x:i for i,x in enumerate(inputs_str)}
    for i,inp in enumerate(inputs_str):
        for n in neigh([int(l) for l in inp],1):
            n_str = "".join([str(x) for x in n])
            if f[inp_dict[inp]] != f[inp_dict[n_str]]:
                e+=1
                break
    return e/len(inputs_str)

from sympy import symbols
from sympy.logic.boolalg import SOPform, POSform
# dontcares = [[float(l) for l in "{0:07b}".format(i)] for i in range(0,2**input_dim) if not (calc_KC("{0:07b}".format(i)) < 10)]
'''
ASSUMES n=7, mainly for the symbols bit
'''
# inputs = [[float(l) for l in "{0:07b}".format(i)] for i in range(0,2**7)]
#dontcares = [x for x in full_inputs if x not in inputs]
def bool_complexity(inputs,ttable):
        dontcares = []
        x1,x2,x3,x4,x5,x6,x7=symbols('x1 x2 x3 x4 x5 x6 x7')
        constraints=[inputs[i] for i in range(len(inputs)) if ttable[i] == '1']
        circuit1=SOPform([x1,x2,x3,x4,x5,x6,x7], constraints, dontcares=dontcares)
        circuit2=POSform([x1,x2,x3,x4,x5,x6,x7], constraints, dontcares=dontcares)
        return min(circuit1.count_ops(), circuit2.count_ops())
# inputs[0]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib

from scipy.special import binom

m = 30
p = lambda x,k: binom(m,k)*(x**k)*((1-x)**(m-k))
g = lambda x,n: np.sum([p(x,k) for k in range(n+1)])
g(0.1,0)

xrange = np.arange(0,1,0.01)
prior = lambda x: 10 if (x<0.6 and x>0.4) else 1
S = np.sum([prior(x) for x in xrange])
priormemo = {x: prior(x)/S for x in xrange}
prior = lambda x: priormemo[x]
N = lambda k: np.sum([p(x,k)*prior(x) for x in xrange])
Nmemo = {k:N(k) for k in range(m+1)}
N = lambda k: Nmemo[k]
N(2)
pp = lambda x,k: (p(x,k)*prior(x))/N(k)

cumpp = lambda i,k: np.sum([pp(y,k) for y in xrange[:i]])
cumpp2 = lambda i,k: np.sum([pp(y,k) for y in xrange[i+1:]])

fbayes = lambda k: np.min([x for i,x in enumerate(xrange) if cumpp2(i,k)<d])
fbayesmemo = {k:fbayes(k) for k in range(m+1)}
fbayes = lambda k: fbayesmemo[k]

plt.scatter(range(m),list(map(f,range(m))))
plt.scatter(range(m),list(map(fbayes,range(m))))

fbayesdelta = lambda c: np.sum([p(c,k) for k in range(m) if fbayes(k) < c])
np.max([fbayesdelta(c) for c in np.linspace(0,1,1000)])

d=0.01
f = lambda k: np.max([x for x in np.linspace(0,1,1000) if g(x,k)>=d])
fmemo = {k:f(k) for k in range(m+1)}
f = lambda k: fmemo[k]
f(0)

plt.scatter(range(m),list(map(f,range(m))))

5/30

np.sum([p(0.49,k)*f(k) for k in range(m+1)])
np.sum([p(0.49,k)*f1(k) for k in range(m+1)])

def f1(k):
    if k==6:
        return f(6)-0.01
    if k==5:
        return f(6)
    else:
        return f(k)
c=0.4
[k for k in range(m) if f(k)<c]

fdelta = lambda c: np.sum([p(c,k) for k in range(m) if f(k) < c])
fdelta(0.1)
np.max([fdelta(c) for c in np.linspace(0,1,100)])
fdelta(0.797979)
np.linspace(0,1,100)[79]

plt.scatter(range(m),list(map(f1,range(m))))

f(m)
f1(6)
p(f1(6)-0.002,6)
[p(f(k),k) for k in range(m+1)]

f1delta = lambda c: np.sum([p(c,k) for k in range(m) if f1(k) < c])
np.max([f1delta(c) for c in np.linspace(0,1,1000)])
np.linspace(0,1,1000)[np.argmax([f1delta(c) for c in np.linspace(0,1,1000)])]

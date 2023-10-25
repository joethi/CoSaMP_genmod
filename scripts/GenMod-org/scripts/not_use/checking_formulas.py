import numpy as np
from scipy.special import factorial
import scipy.optimize as opt

import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


p=10
d=5000
k=2*d+1
P=float(ncr(p+d,p))
L=p*k*np.sqrt(P)
r=np.sqrt(k)/2
delta=0.1
c0 = 1/40
c1 = 10000
c3 = 48*np.log(48)**3/(50*np.log(50)**3)
c2 = c3/c1
gamma = 30


J = np.log(P**(1/2)*3**(p/2))/np.log(2)
a_Jk = 2*(np.log(5) + 4*J*k)/c3
maxterm1 = 1/c0*(np.log(a_Jk/c0))**3*np.log(P)
maxterm2 = 1/gamma*np.exp((a_Jk/(gamma*np.log(P)))**(1/3))
g = np.maximum(maxterm1,maxterm2)

alpha = .2
N_bound_1 = 3**(p+1)/c2*k*np.log(4*L*r/delta)*g/(alpha**2)
N_bound_1
P-N_bound_1
N_bound_1/P

N_bound_2_aa = 3**(p+1)/c2*k*np.log(4*L*r/delta)*np.log(P)**4/c0/(alpha**2)

N_bound_3a = 3**(p+1)/c2*k*np.log(4*L*r/delta)*3**p*np.log(P)/gamma/(alpha**4)*16
N_bound_3b = 3**(p+1)/c2*k*np.log(4*L*r/delta)*np.log(P)*np.log(3**p*np.log(P)/(alpha/4)**2)**3/c0/(alpha**2)

3**p*alpha**-2*np.log(P)

epsi = .2
s_eq = opt.brentq(lambda x: N_bound_1*epsi**2 - c1*3**p*x*np.log(x)**3*np.log(P),1,10**20)
s_eq


P=100000000
p=2
N=100000000
A = c0/gamma
B = N/(c1*3**p)

epsilon0 = opt.brentq(lambda x: np.exp((A*B/(np.log(P))**2 * x**2)**(1/6))-(B/A * x**2)**1/2,10**-3,10**3)

c0=1
gamma=1
P=10
s_e1 = opt.brentq(lambda x: x - gamma*np.log(P)**2*(np.log(x))**2/c0, 10, 10**12)

gamma/c0*np.log(P)**2

2*(np.log(2)**2)

P*c0/gamma/np.log(P)**2

x=np.arange(1,10,.1)
x=np.exp(1)
y=x - gamma*np.log(P)**2*(np.log(x))**2/c0

fig = go.Figure()
fig.add_trace(go.Scatter(x=x,y=y))
fig.show()


epsilon = 16*3**p*np.log(P)
s_e2 = opt.brentq(lambda x: N*epsilon**2 - c1*3**p*x*np.log(x)**3*np.log(P),1,10**5)


epsilonmin = opt.brentq(lambda x: N*x**2 - c1*3**p*(16*3**p*x**(-2)*np.log(P))*np.log(16*3**p*x**(-2)*np.log(P))**3*np.log(P),1,10**5)

np.sqrt(16*3**p*np.log(P))

np.sqrt(np.sqrt(c1*16*3**(2*p)*np.log(P)/N))

P = np.arange(100,1000000,1)
l=100
k=10
L=1
r=1

delta=1/4
p=4
d=np.arange(100,10000,1)
k=2*d+1
#l=d
P = np.zeros(np.size(d))
for i in range(np.size(d)):
    print(i)
    P[i]=float(ncr(p+d[i],p))
N=np.log(P)**4*np.exp(3**(p/4)*(k*np.log(L*r/delta)/np.log(P)**2+l*np.log(P)/np.log(P)**2))


fig = go.Figure()
fig.add_trace(go.Scatter(x=P,y=N))
fig.show()


16*3**p*epsilonmin**(-2)*np.log(P)*epsilonmin**2


np.sqrt(16*3**p*(1000)**(-2)*np.log(P))



1-P**(-gamma*np.log(s_eq)**3)

prop = 1 - 4*np.exp(-alpha**2*c2/3**(p+1)/g*N_bound_1)

h = c1*np.log(22)**3/(22*c0**2*c3**2)*3**p*np.log(P)
N_bound_2 = (3*k*np.log(4*L*r/delta))**2*h/(alpha**2)


hb = 3**p*P/(c1*np.log(P)**4*c2**2*gamma**2)
N_bound_2_b = (3*k*np.log(4*L*r/delta))**2*hb/(alpha**2)


#Might be a good one
N_bound_2_c = c1*3**p*np.log(P)**4*np.exp(((3*k*np.log(4*L*r/delta))/(np.log(P)*gamma))**(1/3))/(alpha**2)

P
P - N_bound_2_c
P - N_bound_2
P - N_bound_2_aa
P - N_bound_1

#Given bound what is the sparsity
#h = c0*c2*np.sqrt(c1*np.exp(1)**3/(27*3**p*np.log(P)))
fN_3 = np.sqrt(N_bound_2/h)
epsilon_max = (np.log(5) + 4*J*k)/fN_3 + 1
s_max = opt.brentq(lambda x: N_bound_2*epsilon_max**2 - c1*3**p*x*np.log(x)**3*np.log(P),1,10**20)

s_min = opt.brentq(lambda x: N_bound_1*alpha**2 - c1*3**p*x*np.log(x)**3*np.log(P),1,10**20)

3**p*alpha**-2*np.log(P)

#Is s_max within correct region for using alternative bound
c0*s_max - gamma*(np.log(s_max)**3*np.log(P)) < 0


c0/(np.log(P)*np.log(smax)**3) - gamma/smax

##
N1 = 1000000
s1 = opt.brentq(lambda x: N1 - c1*3**p*x*np.log(x)**3*np.log(P),1,10**20)

smax = opt.brentq(lambda x: c0/(np.log(P)*np.log(x)**3) - gamma/x,100,10**20)
h2 = smax*gamma*3**p/c0**2/c2
N_bound_3 = (3*k*np.log(4*L*r/delta))*h2/(alpha**2)

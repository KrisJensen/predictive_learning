#%%

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import time
import os
import pickle
from perceptron_utils import *


#%%
N = 5000
rho = 0.3
sig_ss = 0.8
sig_tt = 1.0

z0 = np.random.normal(0, 1, N)
z1 = np.random.normal(0, 1, N)

wstar = z0
w = rho * z0 + np.sqrt(1-rho**2) * z1

w = sig_ss * w / np.sqrt(np.square(w).sum())
wstar = sig_tt * wstar / np.sqrt(np.square(wstar).sum())
rho_emp = np.sum(w*wstar) / (np.sqrt(np.sum(w**2)) * np.sqrt(np.sum(wstar**2)))

print([rho, rho_emp.item()], [[np.sum(w1*w2).item() for w1 in [w, wstar]] for w2 in [w, wstar]])

pg = calc_pg(rho)
pg_sim = np.mean([sign(w @ X) == sign(wstar @ X) for X in [np.random.normal(0, 1, (N, 20240))]])

print(pg, pg_sim)

#%%

K = 50001
pc = 0.75

X = np.random.normal(0, 1, (N, K))

z = w @ X
y = np.sign(z)
ystar = np.sign(wstar @ X)

mask = np.random.binomial(1, p = pc, size = K) #np.random.choice([0, 1], K, p = [1-pc, pc])
ytarget = mask*ystar - (1-mask)*ystar

print(np.mean(ystar == ytarget), ytarget.mean())

#%%

dw = ((0.5 - theta(z) + 0.5*ytarget)[None, :] * X).mean(-1)

sig_ss = np.sqrt(np.sum(w*w))
sig_tt = np.sqrt(np.sum(wstar*wstar))
sig_st = np.sqrt(np.sum(w*wstar))

print(np.sum(dw * wstar), beta*((2*pc - 1)*sig_tt - sig_st**2/sig_ss))
print(np.sum(dw * w), beta*((2*pc - 1)*sig_st**2/sig_tt - sig_ss))

#%%

for M in [1,2,5]:
    for T in [1,2,5]:
        for pg in [0.5, 0.6, 0.7, 0.8, 0.9, 0.9999]:
            pred_pc, pred_goodseq = calc_pc(pg, M, T), calc_p_good_seq(pg, M, T)
            sim_pc, sim_goodseq = calc_pc_emp(pg, M, T, K = 100000)
            print(f"M={M}, T={T}, pg={pg}, good seq = {[np.round(x, 3).item() for x in [pred_goodseq, sim_goodseq]]}, pc = {[np.round(x, 3).item() for x in [pred_pc, sim_pc]]}")


# %%

v = w

proj = (theta(v @ X)[None, :] * X).mean(-1)
beta_emp = np.sqrt(np.square(proj).sum())

print(beta_emp, beta, pearsonr(proj, v))




# %%


N = 4000
rho = 0.0
sig_ss = 1.0
sig_tt = 1.0

z0 = np.random.normal(0, 1, N)
z1 = np.random.normal(0, 1, N)

v = z0
u = rho * z0 + np.sqrt(1-rho**2) * z1

v = sig_ss * v / np.sqrt(np.square(v).sum())
u = sig_tt * u / np.sqrt(np.square(u).sum())

kappa = (1 + 2/pi*rho*np.arcsin(rho)+2/pi*np.sqrt(1-rho**2))/(1+2/pi*np.arcsin(rho))
anal = kappa*v

K = 50000
X = np.random.normal(0, 1, (N, K))

a = v @ X # K
b = u @ X # K

cond = (np.sign(a)==np.sign(b))
emp = (a[None, cond]*X[:, cond]).mean(-1)

#emp =  ((cond * a)[None, :] * X).sum(-1) /  cond.sum()

print(pearsonr(anal, emp))
print(np.sum(anal**2), np.sum(emp**2))
print(pearsonr(v, emp))
print(pearsonr(u, emp))
print(pearsonr(u, v))

#%%

cond1 = (a > 0) & (b > 0)
cond2 = (a < 0) & (b < 0)

x1 = (a[None, cond1]*X[:, cond1]).mean(-1)
x2 = (a[None, cond2]*X[:, cond2]).mean(-1)

print(pearsonr(x1, x2))
for x in [x1, x2]:
    print(pearsonr(v, x))
    print(pearsonr(u, x))

# %%
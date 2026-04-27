#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

D = 1000
N = 10
pg = 0.5
T = 10
phi = lambda z: 1/(1+np.exp(-z))
theta = lambda z: np.float64(z > 0.0)
sgn = lambda z: np.float64(np.sign(z))

wstar = np.random.normal(0, 1, D)/np.sqrt(D)
w = np.random.normal(0, 1, D)/np.sqrt(D)
error = wstar - w

beta = np.abs(np.random.normal(0, 1, 10000)).mean()

#%% some functions

def pc_f(N, T, pg):
    pc = 1 - (1 - pg**T)**N
    pc += (1 - pg**T)**N * pg * (1-pg**T - (1-pg)**T)/(1 - pg**T)
    return pc

def dW(X, target, w):
    #return ((target - phi(w @ X))[:, None, :] * X).mean(-1)
    return ((target - theta(w @ X))[:, None, :] * X).mean(-1)

#%%

K = 10001
X = np.random.normal(0, 1, (K, D, T))

Ystar = theta(wstar @ X)

dW_sup = dW(X, Ystar, w)

#%%

print(pearsonr(dW_sup.mean(0), error)[0], dW_sup.std(0).mean(0))
print((dW_sup * error[None, :]).sum(-1).mean())

# %%

sims_correct = np.random.binomial(1, p = pg, size = (K, N, T))
Ysims = sims_correct * Ystar[:, None, :] + (1-sims_correct)*(1-Ystar[:, None, :])

scores = (Ysims == Ystar[:, None, :]).sum(-1)
max_scores = np.amax(scores, -1)
max_inds = np.argmax(scores, -1)
max_inds[max_scores < T-0.5] = 0 # just pick first if it wasn't entirely correct

Ytarget = Ysims[np.arange(K), max_inds, :]

print(pc_f(N, T, pg), (Ytarget == Ystar).mean())

# %%

dW_sim = dW(X, Ytarget, w)
print(pearsonr(dW_sim.mean(0), wstar - w)[0], dW_sim.std(0).mean(0))
print((dW_sim * error[None, :]).sum(-1).mean())

# %%

pgs = np.array([0.5, 0.55, 0.6, 0.65, 0.7])
Ts = np.arange(1, 11, dtype = float)[::-1]
Ns = np.arange(1, 22, 2, dtype = float)

data = pc_f(Ns[None, None, :], Ts[None, :, None], pgs[:, None, None])

fig, axs = plt.subplots(1, len(pgs), figsize = (8,3))
for ipg, ax in enumerate(axs):
    ax.imshow(data[ipg], vmin = 0.5, vmax = 1)
    if ipg == 0:
        ax.set_ylabel("T")
        ax.set_yticks(np.arange(len(Ts))[::2], Ts[::2].astype(int))
    else:
        ax.set_yticks([])
    ax.set_xticks(np.arange(len(Ns))[::2], Ns[::2].astype(int))
    ax.set_title(pgs[ipg])
ax.set_xlabel("num sims")
plt.show()
plt.close()



# %%

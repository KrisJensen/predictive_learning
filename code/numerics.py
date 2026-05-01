#%%

import numpy as np

phi = lambda x: 1/(1+np.exp(-x))

zs = np.random.normal(0,1,10000000)
sigs = phi(zs)
print(np.sqrt(np.mean((sigs*(1-sigs))**2)))


print(np.mean(sigs))

print(np.mean(sigs**2))

#%%

ys = np.random.binomial(1, p = sigs)*2.0-1.0
print(np.mean(ys*sigs))

# %%

T = 5

pg = 0.6

for T in range(1, 10):
    samps = np.random.choice([0, 1], p = [1-pg, pg], replace = True, size = (10000000, T))

    samps = samps[samps.sum(-1) < T-0.5]
    pred = pg*(1-pg**(T-1))/(1-pg**(T))

    #print(f"T={T}, prediction={np.round(np.mean(samps), 4)}, sim={np.round(pg*(1-pg**T - (1-pg)**T)/(1-pg**(T)), 4)}")
    print(f"T={T}, prediction={np.round(pred, 4)}, sim={np.round(np.mean(samps), 4)}")


# %%


def Pc(T, K):
    """Probability that I get a correct sequence of length T in K samples"""
    return 1-(1-2**(-T))**K

def Ps(T):
    """Probability that a single sequence element is correct given that at least one of the T elements is incorrect"""
    return 0.5*(1-2**(1-T))/(1-2**(-T))


K = 10
for T in range(1,20, 2):
    # probability that I update towards the opposite of my target
    print((1-Pc(T, K))*(1-Ps(T)))

# %%


#%%

import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
tab10 = plt.get_cmap("tab10")

#%%

Nin = 1000
T = 6
phi = lambda x: 1/(1+np.exp(-x))

Wteach = np.random.normal(0, 1, (1, Nin))/np.sqrt(Nin)
Wstud = np.random.normal(0, 1, (1, Nin))/np.sqrt(Nin)

def get_output(W, X, argmax = False):
    pi = phi(W @ X) # S x 1 x T
    if argmax:
        y = np.sign(pi - 0.5)
    else:
        y = np.random.binomial(1, p = pi)*2.0-1.0
    return pi, y

def test(teach, stud, gamma, samples = 1000):

    X = np.random.normal(0, 1, (samples, Nin, T)) # S x N x T
    yteach = np.sign(teach @ X) # S x 1 x T
    pi, ystud = get_output(stud, X, argmax = True)

    corrects = np.mean(ystud == yteach)
    outputs = np.mean(ystud)
    confs = np.mean(2*np.abs(pi - 0.5))
    rew = rewfunc(ystud, yteach, gamma = gamma)

    return corrects, outputs, confs, rew.mean()

#%%
print(test(Wteach, Wstud, gamma = 1.0))

def rewfunc(ystud, yteach, gamma = 1.0):

    partial = np.mean(ystud == yteach, axis = (-1, -2)) # each one (batch, ...)
    all = np.all(ystud == yteach, axis = (-1, -2)).astype(float) # all correct # (batch, ...)

    return gamma*partial + (1-gamma)*all


#%% now develop some simple heuristic target
# do so in 1 of two ways:
#   simples samples with a heuristic
#   bit flips correlated with the truth

def sample_target(pi, base, target, nsamps = 10, gamma = 1.0):
    """
    pi: policy
    base: initial action sequence
    target: true action sequence
    nsamps: number of samples to evaluate from the policy
    """

    guesses = np.random.binomial(1, p = pi, size = (nsamps,)+pi.shape)*2.0-1.0 # (nsamps, batch, 1, T)
    rews = rewfunc(guesses, target, gamma = gamma) # (nsamps, batch)
    best = guesses[np.argmax(rews, axis = 0), np.arange(rews.shape[-1]), ...]
    #print(rewfunc(base, target, gamma = gamma).mean(), rewfunc(best, target, gamma = gamma).mean())
    return best

    #rew = rewfunc(base, target, gamma = gamma)
    #print(rew.mean())
    # for _ in range(nsamps):
    #     guess = np.random.binomial(1, p = pi)*2.0-1.0 # (batch, 1, T)
    #     new_rew = rewfunc(guess, target, gamma = gamma) # (batch, )
    #     better_inds = np.where(new_rew > rew) #
    #     base[better_inds] = guess[better_inds]
    #     rew[better_inds] = new_rew[better_inds]

    #print(rew.mean())
    return base


def flip_target(pi, base, target, flip_prob = 0.2):
    """
    pi: policy
    base: initial action sequence
    target: true action sequence
    flip_prob: probability of flipping an incorrect bit
    """

    flip_probs = flip_prob * (base != target) # probability of flipping error bits
    flips = np.random.binomial(1, p = flip_probs).astype(bool)

    #print(np.mean(base == target))
    base[flips] = -base[flips]
    #print(np.mean(base == target))

    return base

# %%

def calc_grad(X, pi, ystud, yteach, rew, T, gamma = 1.0, mode = "RL", target = "true", nsamps = 10, flip_prob = 0.3, baseline = None, **kwargs):

    assert mode in ["RL", "supervised"]
    assert target in ["true", "sample", "flip"]

    if mode == "RL":
        # dJ/dW = \sum_t (R-B)dpi/dW
        # pi(a = 1) = 1/(1+e^-Wx)
        # dpi(a=1)/dWx = pi(1-pi)
        # dpi(a = -1)/dWx = -pi(1-pi)
        if baseline is None:
            baseline = gamma*0.5 + (1-gamma)*0.5**T # accuracy of random agent
            
        dLdW = T*2*(rew[:, None, None]-baseline) * ystud*pi*(1-pi)*X #batch x N x T

    elif mode == "supervised":
        # supervised
        if target == "true":
            ytarget = yteach
        elif target == "sample":
            ytarget = sample_target(pi, ystud, yteach, nsamps = nsamps, gamma = gamma)
        elif target == "flip":
            ytarget = flip_target(pi, ystud, yteach, flip_prob = flip_prob)

        dLdW = ytarget*pi*(1-pi)*X # batch x N x T
    
    return dLdW

def run_sim(Nin = 1000, T = 6, gamma = 1.0, epochs = 15000, eval_every = 500, lr = 1e-2,
            reg = 1e-3, batch_size = 5, eval_num = 5000, Print = False, adaptive_baseline = False,
            **kwargs):

    Wteach = np.random.normal(0, 1, (1, Nin))/np.sqrt(Nin)
    Wstud = np.random.normal(0, 1, (1, Nin))/np.sqrt(Nin)
    if adaptive_baseline:
        baseline = test(Wteach, Wstud, gamma = gamma)[-1]
    else:
        baseline = None

    accs = []
    for epoch in range(epochs):

        X = np.random.normal(0, 1, (batch_size, Nin, T)) # batch x N x T
        yteach = np.sign(Wteach @ X) # batch x 1 x T
        pi, ystud = get_output(Wstud, X, argmax = False) # batch x 1 x T
        rew = rewfunc(ystud, yteach, gamma = gamma) # (batch, )
        
        dLdW = calc_grad(X, pi, ystud, yteach, rew, T, gamma = gamma, baseline = baseline, **kwargs) # (batch, N, T)
        delta_W = np.mean(dLdW, axis = (0, -1))[None, :] # 1 x N
        Wstud += lr * (delta_W - reg*Wstud)

        if epoch % eval_every == 0:
            corrects, outputs, confs, rew = test(Wteach, Wstud, gamma = gamma)
            accs.append(corrects)
            if Print:
                print(epoch, corrects, outputs, confs, rew)
            if adaptive_baseline:
                baseline = rew

    return accs


#%% test some different models

#abc = run_sim(T=5, mode = "supervised", target = "sample", nsamps = 50, gamma = 0.0, Print = True)

#abc = run_sim(T=5, mode = "supervised", target = "flip", flip_prob = 0.5, Print = True)

abc = run_sim(T=10, mode = "RL", gamma = 1.0, Print = True, adaptive_baseline = True)

#%%
Ts = [1,4,7,10]
models = [{"name": "RL_1", "mode": "RL", "gamma": 1.0},
            {"name": "RL_0", "mode": "RL", "gamma": 0.0},
            {"name": "sup", "mode": "supervised", "target": "true"},
            #{"name": "sample3_1", "mode": "supervised", "target": "sample", "nsamps": 3, "gamma": 1.0},
            {"name": "sample10_1", "mode": "supervised", "target": "sample", "nsamps": 10, "gamma": 1.0},
            {"name": "sample10_0", "mode": "supervised", "target": "sample", "nsamps": 10, "gamma": 0.0},
            {"name": "sample100_0", "mode": "supervised", "target": "sample", "nsamps": 100, "gamma": 0.0},
            #{"name": "flip0.3", "mode": "supervised", "target": "flip", "flip_prob": 0.3},
            {"name": "flip0.5", "mode": "supervised", "target": "flip", "flip_prob": 0.5}]

all_accs = []
for model in models:
    all_accs.append([])
    for iT, T in enumerate(Ts):
        all_accs[-1].append(run_sim(T=T, **model))
        print(model, T, all_accs[-1][-1][-1])
all_accs = np.array(all_accs)


#%% plots

# make a plot across all T just for the three base models
plt.figure()
for i in [0,1,2]:
    for iT, T in enumerate(Ts):
        col = np.array(tab10(i))
        col[:3] *= (7-iT)/ (7)
        label = models[i]["name"] if iT == 0 else None
        plt.plot(all_accs[i, iT, :], ls = "-", color = col, label = label)
plt.legend()
plt.savefig(f"./figs/base_model_learning.png", bbox_inches = "tight")
plt.show()


#%% make a plot across all models for each T
for iT, T in enumerate(Ts):
    plt.figure()
    for imodel, model in enumerate(models):
        plt.plot(all_accs[imodel, iT, :], color = tab10(imodel), ls = "-", label = model["name"])
    plt.legend()
    plt.ylim(0.49, 1.0)
    plt.title(f"T = {T}")
    plt.savefig(f"./figs/model_learning_T{T}.png", bbox_inches = "tight")
    plt.show()


#%% compute gradient mean and variance

batch_size = 20000
Nin = 1000
Wteach = np.random.normal(0, 1, (1, Nin))/np.sqrt(Nin)
Wstud = np.random.normal(0, 1, (1, Nin))/np.sqrt(Nin)

Ts = [1,2,3,5,7,10]
res = np.zeros((len(models), len(Ts), 3))
for imodel, model in enumerate(models):
    gamma = model["gamma"] if "gamma" in model.keys() else 1.0
    print()
    for iT, T in enumerate(Ts):
        X = np.random.normal(0, 1, (batch_size, Nin, T)) # batch x N x T
        yteach = np.sign(Wteach @ X) # batch x 1 x T
        pi, ystud = get_output(Wstud, X, argmax = False) # batch x 1 x T
        rew = rewfunc(ystud, yteach, gamma = gamma) # (batch, )

        dLdW = calc_grad(X, pi, ystud, yteach, rew, T, **model) # (batch, N, T)
        grad = dLdW.mean(-1) # (batch, N)

        mean_grad = np.mean(grad, axis = 0) # mean gradient across batches (N,)
        corr = pearsonr(mean_grad, (Wteach-Wstud).flatten())[0] # correlation with true error

        mean = np.abs(np.mean(grad, axis = 0)).mean() # mean across batches for each parameter, then mean across parameters of the abs
        std = np.std(grad, axis = 0).mean() # std across batches for each parameter, then mean across parameters
        res[imodel, iT, :] = np.array([mean, std, corr])

        print(model["name"], T, ":", std, mean, std/mean, corr)
        

#%%
ratio = res[..., 1]/res[..., 0]
plt.figure()
for imodel, model in enumerate(models):
    plt.plot(Ts, ratio[imodel, :], color = tab10(imodel), label = model["name"])
plt.ylim(0, np.amax(ratio)*1.05)
plt.legend()
plt.xlabel("T")
plt.ylabel("normalised std")
plt.savefig(f"./figs/initial_gradient_noise.png", bbox_inches = "tight")
plt.show()


corrs = res[..., -1]
plt.figure()
for imodel, model in enumerate(models):
    plt.plot(Ts, corrs[imodel, :], color = tab10(imodel), label = model["name"])
plt.ylim(0.0, 1.05*corrs.max())
plt.legend()
plt.xlabel("T")
plt.savefig(f"./figs/initial_gradient_bias.png", bbox_inches = "tight")
plt.show()

# %%




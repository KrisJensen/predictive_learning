
#%%

import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
tab10 = plt.get_cmap("tab10")


#%%


def rewfunc(ystud, yteach, gamma = 1.0):

    partial = np.mean(ystud == yteach, axis = (-1, -2)) # each one (batch, ...)
    all = np.all(ystud == yteach, axis = (-1, -2)).astype(float) # all correct # (batch, ...)

    return gamma*partial + (1-gamma)*all

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

def calc_olap(teach, stud):
    return np.mean(np.sum((teach - teach.mean()) / np.sqrt(np.square(teach).sum()) * (stud - stud.mean()) / np.sqrt(np.square(stud).sum()) ))

def test(teach, stud, gamma, samples = 1000):

    X = np.random.normal(0, 1, (samples, Nin, T)) # S x N x T
    yteach = np.sign(teach @ X) # S x 1 x T
    pi, ystud = get_output(stud, X, argmax = True)

    corrects = np.mean(ystud == yteach)
    outputs = np.mean(ystud)
    confs = np.mean(2*np.abs(pi - 0.5))
    rew = rewfunc(ystud, yteach, gamma = gamma)
    olap = calc_olap(teach, stud)
    return corrects, outputs, confs, rew.mean(), olap

print(test(Wteach, Wstud, gamma = 1.0))

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

def calc_grad(X, pi, ystud, yteach, rew, T, log = False, gamma = 1.0, mode = "RL", target = "true", nsamps = 10, flip_prob = 0.3, baseline = None, **kwargs):

    assert mode in ["RL", "supervised"]
    assert target in ["true", "sample", "flip"]

    if mode == "RL":
        # dJ/dW = \sum_t (R-B)d log pi/dW
        # pi(a = 1) = 1/(1+e^-Wx)
        # d log pi(a=1)/dWx = (1-pi)
        # dpi(a = -1)/dWx = -pi
        ytarget = ystud
        if baseline is None:
            baseline = gamma*0.5 + (1-gamma)*0.5**T # accuracy of random agent

        #dLdW = T*2*(rew[:, None, None]-baseline) * ystud*pi*(1-pi)*X #batch x N x T
        
        # We did our maths with the _summed_ reward but here compute the _mean_ reward.
        # we scale by T here to make up for this (this makes the expected gradient independent of T)
        scale = gamma * T + (1-gamma)*T
        
        #scale = gamma * T + (1-gamma)*2**(T-1) # this is the correct lr only at initialisation
        dLdW = scale*(rew[:, None, None]-baseline) * (0.5 - pi + 0.5*ystud)*X #batch x N x T

    elif mode == "supervised":
        # supervised
        if target == "true":
            ytarget = yteach
        elif target == "sample":
            ytarget = sample_target(pi, ystud, yteach, nsamps = nsamps, gamma = gamma)
        elif target == "flip":
            ytarget = flip_target(pi, ystud, yteach, flip_prob = flip_prob)

        if log:
            dLdW = (0.5 - pi + 0.5*ytarget)*X # batch x N x T
        else:
            dLdW = ytarget*pi*(1-pi)*X # batch x N x T

    
    return dLdW, ytarget

def run_sim(Nin = 1000, T = 6, gamma = 1.0, epochs = 40000, eval_every = 2000, lr = 5e-3,
            reg = 1e-3, batch_size = 1, log = True, eval_num = 5000, Print = False, adaptive_baseline = False,
            **kwargs):

    Wteach = np.random.normal(0, 1, (1, Nin))/np.sqrt(Nin)
    Wstud = np.random.normal(0, 1, (1, Nin))/np.sqrt(Nin)
    if adaptive_baseline:
        baseline = test(Wteach, Wstud, gamma = gamma)[-1]
    else:
        baseline = None

    accs, olaps, rews  = [], [], np.zeros((epochs, batch_size))
    ystuds, yteachs, ytargets, pis = [np.zeros((epochs, batch_size, 1, T)) for _ in range(4)]
    for epoch in range(epochs):

        X = np.random.normal(0, 1, (batch_size, Nin, T)) # batch x N x T
        yteach = np.sign(Wteach @ X) # batch x 1 x T
        pi, ystud = get_output(Wstud, X, argmax = False) # batch x 1 x T
        rew = rewfunc(ystud, yteach, gamma = gamma) # (batch, )
        
        dLdW, ytarget = calc_grad(X, pi, ystud, yteach, rew, T, gamma = gamma, baseline = baseline, **kwargs) # (batch, N, T)
        delta_W = np.mean(dLdW, axis = (0, -1))[None, :] # 1 x N
        Wstud += lr * (delta_W - reg*Wstud)

        rews[epoch], ystuds[epoch], yteachs[epoch], ytargets[epoch], pis[epoch] = rew, ystud, yteach, ytarget, pi
        if epoch % eval_every == 0:
            corrects, outputs, confs, rew, olap = test(Wteach, Wstud, gamma = gamma)
            accs.append(corrects)
            olaps.append(olap)
            if Print:
                print(epoch, corrects, outputs, confs, rew, olap)
            if adaptive_baseline:
                baseline = rew

    return {"accs": accs, "rews": rews, "ystuds": ystuds, "yteachs": yteachs, "ytargets": ytargets, "pis": pis, "olaps": olaps}


#%% test some different models

#abc = run_sim(T=7, mode = "supervised", gamma = 1.0, Print = True, log = True)

abc = run_sim(T=10, mode = "supervised", gamma = 0.0, target = "sample", nsamps = 100, Print = True, log = True, lr = 10e-3)

#abc = run_sim(T=5, mode = "supervised", target = "flip", flip_prob = 0.5, Print = True)

#abc = run_sim(T=10, mode = "RL", gamma = 1.0, Print = True, adaptive_baseline = True)

#%%
np.random.seed(1)
gamma, epochs = 1.0, 10000
gamma, epochs = 0.0, 80000
res1 = run_sim(T=9, mode = "supervised", gamma = gamma, target = "sample", nsamps = 100, Print = True, log = True, lr = 1e-2, epochs = epochs)
res2 = run_sim(T=9, mode = "RL", gamma = gamma, Print = True, log = True, lr = 1e-2, epochs = epochs)
#%%

for gamma in [0.0, 1.0]:
    plt.figure()
    for ires, res in enumerate([res1, res2]):
        rews = rewfunc(res["ystuds"], res["yteachs"], gamma = gamma).flatten()
        grews = rewfunc(np.sign(res["pis"] - 0.5), res["yteachs"], gamma = gamma).flatten()
        trews = rewfunc(res["ytargets"], res["yteachs"], gamma = gamma).flatten()
        for irew, rew in enumerate([rews, grews, trews]):
            plt.plot(gaussian_filter1d(rew, 1000), color = plt.get_cmap("tab10")(irew), ls = ["-", "--"][ires])
    plt.legend(["student", "greedy", "target"])
    plt.show()

plt.figure()
plt.plot(res1["olaps"], ls = "-", color = "k")
plt.plot(res2["olaps"], color = "k", ls = "--")
plt.show()


#%%

# these are the models we want to look at
models = [{"name": "RL_1", "mode": "RL", "gamma": 1.0},
            {"name": "RL_1_adapt", "mode": "RL", "gamma": 1.0, "adaptive_baseline": True},
            {"name": "RL_0", "mode": "RL", "gamma": 0.0},
            {"name": "sup", "mode": "supervised", "target": "true", "log": False},
            {"name": "logsup", "mode": "supervised", "target": "true", "log": True},
            #{"name": "sample3_1", "mode": "supervised", "target": "sample", "nsamps": 3, "gamma": 1.0},
            {"name": "sample10_1", "mode": "supervised", "target": "sample", "nsamps": 10, "gamma": 1.0},
            {"name": "sample10_0", "mode": "supervised", "target": "sample", "nsamps": 10, "gamma": 0.0},
            {"name": "sample200_0", "mode": "supervised", "target": "sample", "nsamps": 200, "gamma": 0.0},
            #{"name": "flip0.3", "mode": "supervised", "target": "flip", "flip_prob": 0.3},
            {"name": "flip0.5", "mode": "supervised", "target": "flip", "flip_prob": 0.5}]


#%%
Ts = [1,4,7,10]
all_accs = []
for model in models:
    all_accs.append([])
    for iT, T in enumerate(Ts):
        all_accs[-1].append(run_sim(T=T, **model)["accs"])
        print(model, T, all_accs[-1][-1][-1])
all_accs = np.array(all_accs)


#%% plots

# make a plot across all T just for the three base models
xs = np.arange(all_accs.shape[-1])*2000
plt.figure()
for i in [0,1,2]:
    for iT, T in enumerate(Ts):
        col = np.array(tab10(i))
        col[:3] *= (7-iT)/ (7)
        label = models[i]["name"] if iT == 0 else None
        plt.plot(xs, all_accs[i, iT, :], ls = "-", color = col, label = label)
plt.legend()
plt.savefig(f"./figs/base_model_learning.png", bbox_inches = "tight")
plt.show()


#%% make a plot across all models for each T
for iT, T in enumerate(Ts):
    plt.figure()
    for imodel, model in enumerate(models):
        plt.plot(xs, all_accs[imodel, iT, :], color = tab10(imodel), ls = "-", label = model["name"])
    plt.legend()
    plt.ylim(0.49, 1.0)
    plt.title(f"T = {T}")
    plt.savefig(f"./figs/model_learning_T{T}.png", bbox_inches = "tight")
    plt.show()


#%% compute gradient mean and variance

batch_size = 50000
Nin = 1000
Wteach = np.random.normal(0, 1, (1, Nin))/np.sqrt(Nin)
Wstud = np.random.normal(0, 1, (1, Nin))/np.sqrt(Nin)

Ts = np.array([1,2,3,5,7,9])
#batch_size, Ts = 5000, np.array([1,2,3,4,5,6,8])
expres = np.zeros((len(models), len(Ts), 3))
for imodel, model in enumerate(models):
    gamma = model["gamma"] if "gamma" in model.keys() else 1.0
    print()
    for iT, T in enumerate(Ts):
        X = np.random.normal(0, 1, (batch_size, Nin, T)) # batch x N x T
        yteach = np.sign(Wteach @ X) # batch x 1 x T
        pi, ystud = get_output(Wstud, X, argmax = False) # batch x 1 x T
        rew = rewfunc(ystud, yteach, gamma = gamma) # (batch, )

        dLdW = calc_grad(X, pi, ystud, yteach, rew, T, **model)[0] # (batch, N, T)
        grad = dLdW.mean(-1) # (batch, N)

        mean_grad = np.mean(grad, axis = 0) # mean gradient across batches (N,)
        corr = pearsonr(mean_grad, (Wteach-Wstud).flatten())[0] # correlation with true error

        #mean = np.abs(np.mean(grad, axis = 0)).mean() # mean across batches for each parameter, then mean across parameters of the abs
        mean = np.mean(grad, axis = 0).std() # mean across batches for each parameter, then std across parameters
        std = np.std(grad, axis = 0).mean() # std across batches for each parameter, then mean across parameters
        expres[imodel, iT, :] = np.array([mean, std, corr])

        print(model["name"], T, ":", std, mean, std/mean, corr)
        

#%%
ratio = expres[..., 1]/expres[..., 0]
plt.figure()
for imodel, model in enumerate(models):
    plt.plot(Ts, ratio[imodel, :], color = tab10(imodel), label = model["name"])
plt.ylim(0, np.amax(ratio)*1.05)
plt.legend()
plt.xlabel("T")
plt.ylabel("normalised std")
plt.savefig(f"./figs/initial_gradient_noise.png", bbox_inches = "tight")
plt.show()


corrs = expres[..., -1]
plt.figure()
for imodel, model in enumerate(models):
    plt.plot(Ts, corrs[imodel, :], color = tab10(imodel), label = model["name"])
plt.ylim(0.0, 1.05*corrs.max())
plt.legend()
plt.xlabel("T")
plt.savefig(f"./figs/initial_gradient_bias.png", bbox_inches = "tight")
plt.show()


#%% also plot gradients with analytical expressions for the ones I know

to_plot = [(0, 1, 0*Ts+np.sqrt(0.207)/2),
            #(2, np.sqrt(0.207 * Ts * (2.0**(-Ts) - 2.0**(-2*Ts)))), # multiply var by T^2 because we scale lr by T, which is like scaling reward by T
            (2, 2**(Ts-1)/Ts, np.sqrt(0.207 * (2.0**(Ts) - 1.0) / (4*Ts))),
            #(2, 1, np.sqrt(0.207/4 * (2.0**Ts - 1))),
            (3, 1, 0.21174848828296944/np.sqrt(Ts)),
            (4, 1, np.sqrt(0.29338375463560357)/np.sqrt(Ts))]

plt.figure()
for index, scale, ref in to_plot:
    plt.scatter(Ts, scale*expres[index, ..., 1], marker = ".", s = 500, color = tab10(index), label = models[index]["name"])
    plt.plot(Ts, ref, color = tab10(index), ls = "--")

plt.legend()
plt.xlabel("T")
plt.ylabel(r"$<std(g_i)>$")
plt.savefig(f"./figs/initial_analytical.png", bbox_inches = "tight")
plt.show()

plt.figure()
for index, scale, ref in to_plot:
    plt.scatter(Ts, scale*expres[index, ..., 0], marker = ".", s = 500, color = tab10(index), label = models[index]["name"])
plt.legend()
plt.xlabel("T")
plt.ylabel(r"$<mean(g_i)>$")
plt.savefig(f"./figs/initial_mean.png", bbox_inches = "tight")
plt.show()


#%%
# plt.figure()
# plt.plot(Ts, res[2, ..., 0])
# plt.show()

# %%




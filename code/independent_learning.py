
#%%
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import time
import os
import pickle
from perceptron_utils import *


# %% try to simulate a learning trajectory

def setup(mode, linear_sig_coeff, rho):

    baseline = None
    if mode == "RL":
        baseline = lambda pR: pR
    if mode == "RL_raw":
        baseline = lambda pR: 0.0
        mode = "RL"

    if linear_sig_coeff == "beta":
        linear_sig_coeff = beta

    sig_ss2 = 1.0
    sig_st2 = rho

    return mode, baseline, linear_sig_coeff, sig_ss2, sig_st2

def run_sim(T, M = 4, iters = int(1e5), eta = 1e-4, linear_sig_coeff = 0.25, mode = "predictive", normalise = False, verbose = False, approx_sigmoid = "step", rho = 0.0, orthogonal = False):

    assert approx_sigmoid in ["step", "linear"] # how do we approximate the sigmoid in our gradient
    assert mode in ["predictive", "supervised", "RL", "RL_raw"] # which learning algorithm
    
    mode, baseline, linear_sig_coeff, sig_ss2, sig_st2 = setup(mode, linear_sig_coeff, rho)

    if mode == "RL":
        approx_sigmoid = "linear"
    if approx_sigmoid == "linear":
        w_coeff = linear_sig_coeff
    elif approx_sigmoid == "step":
        w_coeff = beta

    data = []

    for iter_ in range(iters):
        #rho = np.minimum(1-1e-10, sig_st2 / np.sqrt(sig_ss2))

        if normalise:
            scale = np.sqrt(sig_ss2)
            sig_ss2 /= scale**2
            sig_st2 /= scale

        sig_ss = np.sqrt(sig_ss2)

        rho = np.clip(sig_st2 / sig_ss, -1.0, 1.0)
        pg = calc_pg(rho)
        pR = calc_p_good_seq(pg, (M if mode == "predictive" else 1), T)# probability of sampling a correct sequence

        if mode == "RL":
            pc = pg # probability of training on good action is simply probability of action being good

            B = baseline(pR)
            kappa = calc_kappa(rho)
            dw_stud = calc_rl_dw_stud(kappa, pR, w_coeff, B, beta)
            dw_teach = calc_rl_dw_teach(kappa, rho, pR, w_coeff)
        else:
            if mode == "supervised":
                pc = 1 # always train on good action
            elif mode == "predictive":
                pc = calc_pc(pg, M, T) # train on sampled action
            dw_stud = -w_coeff # student coefficient in expected update
            dw_teach = beta*(2*pc - 1) # teacher coefficient in expected update

        if orthogonal: # only learn from orthogonal component
            dsig_ss2 = 0.0 # no change in student magnitude
            dsig_st2 = eta*dw_teach*(1 - rho**2)
        else:
            dsig_st2 = eta*( sig_st2/sig_ss*dw_stud + dw_teach )
            dsig_ss2 = 2*eta*( sig_ss*dw_stud + sig_st2*dw_teach)

        sig_ss2 = max(1e-10, sig_ss2+dsig_ss2)
        sig_st2 += dsig_st2

        if iter_ % int(np.floor(iters / 100)) == 0:
            data.append([rho, sig_st2, sig_ss, pg, pc, pR, iter_])

    return np.array(data)

def estimate_grad(mode, w, wstar, T, M = None, batch_size = 10001, approx_sigmoid = "step", linear_sig_coeff = 0.25, baseline = None, independent_samples = True, orthogonal = False):
    N = w.shape[0]

    # sample inputs
    X = np.random.normal(0, 1, (batch_size, N, T))
    Z = w @ X
    theta_Z = theta(Z) # batch x T
    sigma_Z = sigma(Z)
    Ytrue = sign(wstar @ X) # batch x T
    sig_ss = np.sqrt(np.sum(w**2))
    pg = calc_pg(np.sum(w*wstar)/sig_ss)

    # how do we approximate our sigmoid in the gradient update?
    if approx_sigmoid == "step":
        sig_Z = theta_Z # approximate sigmoid with step function
    elif approx_sigmoid == "true":
        sig_Z = sigma_Z
    elif approx_sigmoid == "linear":
        sig_Z = 0.5 + linear_sig_coeff * Z / sig_ss # linearise sigmoid

    if mode == "predictive":
        G = 1 # gain of update
        if independent_samples: # cheat with sampling sequences
            Ytarget, pc = sample_independent_ytarget(Ytrue, pg, M)
        else: # sample from our sigmoids
            scale = 0.0+2*rho # temperature for sampling
            Ytarget, pc = sample_sigmoid_ytarget(Ytrue, Z, scale, M)

    elif mode == "supervised":
        G = 1 # gain of update
        pc = 1 # probability of correct target
        Ytarget = Ytrue # target actions

    elif mode == "RL":
        # compute correlations
        pR = pg ** T # probability of sampling a correct sequence
        pc = pg # train on own actions
        Ytarget = theta_Z*2 - 1 # actions the agent took
        R = ((Ytarget == Ytrue).sum(-1) == T).astype(float) # reward if full sequence is good
        G = (R - baseline(pR))[:, None, None] # TD error

    if mode == "supervised_nolog":
        pc = 1
        ptrue = 0.5 + Ytrue*(sigma_Z - 0.5)
        dptrue = Ytrue*sigma_Z*(1-sigma_Z)
        grads = np.zeros((batch_size, N, T))
        for t in range(T):
            not_t = np.arange(T)[np.arange(T) != t]
            grads[..., t] = ((dptrue[:, t]*np.prod(ptrue[:, not_t], axis = -1))[:, None]*X[..., t])
    else:
        # update takes the same form in all cases
        grads = (G * (0.5 - sig_Z + 0.5*Ytarget)[:, None, :] * X) # size (batch, N, T)

    if orthogonal: # project out component along w
        w_norm = w[None, :, None] / np.sqrt(np.sum(w**2)) # normalised student weights (1, N, 1)
        w_olap = np.sum(grads*w_norm, axis = 1, keepdims = True) # component of gradient along normalised student (batch, 1, T)
        grads = grads - w_olap * w_norm # subtract component along w_norm (batch, N, T)

    return grads, pc

def run_emp_sim(T, M=4, N = 1000, iters = int(1e5), eta = 1e-4, batch_size = 1001, mode = "predictive", normalise = False, approx_sigmoid = "step", linear_sig_coeff = 0.25, overwrite = False, independent_samples = True, rho = 0, orthogonal = False, **kwargs):

    assert mode in ["predictive", "supervised", "RL", "RL_raw", "supervised_nolog"]
    assert approx_sigmoid in ["step", "linear", "true"]
    if mode != "predictive":
        M = None

    prms = [(k.item(), locals()[k]) for k in np.sort(list(locals().keys()))]
    prm_str = "_".join([p[0]+"-"+str(p[1]) for p in prms])
    print(prm_str)
    if (f"{prm_str}.p" in os.listdir(f"{basedir}/data/")) and (not overwrite):
        return pickle.load(open(f"{basedir}/data/{prm_str}.p", "rb"))
    
    mode, baseline, linear_sig_coeff, sig_ss2, sig_st2 = setup(mode, linear_sig_coeff, rho)

    w, wstar = sample_w_wstar(N, rho)

    t0 = time.time()
    data = []

    for iter_ in range(iters):
        
        if normalise:
            w = w / np.sqrt(np.sum(np.square(w)))

        # compute correlations
        sig_ss2 = np.sum(w**2)
        sig_st2 = np.sum(w*wstar)
        sig_ss = np.sqrt(sig_ss2)
        rho = sig_st2/sig_ss
        pg = calc_pg(rho)
        pR = calc_p_good_seq(pg, (M if mode == "predictive" else 1), T)# probability of sampling a correct sequence

        # grad is shape (batch, N, T)
        #grads, pc = estimate_grad(w, wstar, batch_size, T, approx_sigmoid, linear_sig_coeff, mode, baseline, independent_samples, orthogonal)
        grads, pc = estimate_grad(mode, w, wstar, T, M, batch_size, approx_sigmoid, linear_sig_coeff, baseline, independent_samples, orthogonal)
        w += eta*grads.mean((0, 2)) # size N

        # rhos, sigs_st, sigs_ss, pgs, pcs 
        if iter_ % int(np.floor(iters / 100)) == 0:
            data.append([rho, sig_st2, sig_ss, pg, pc, pR, iter_])

        if iter_ % int(np.round(iters / 20)) == 0:
            print("\n", iter_, rho, np.round((time.time() - t0)/60, 1))

    pickle.dump(np.array(data), open(f"{basedir}/data/{prm_str}.p", "wb"))

    return np.array(data)

titles = ["student-teacher correlation", "student-teacher overlap", "student magnitude", "p(guess correct)", "p(action correct)", "p(sequence correct)"]

#%%

M = 4
T = 5
eta = 5e-3
normalise = True

# iters = int(2e4)
# datas = [run_sim(T, M = M, iters = iters, eta = eta, mode = "RL", normalise = normalise, approx_sigmoid = "linear", linear_sig_coeff = 0.25),
# run_sim(T, M = M, iters = iters, eta = eta, mode = "RL", normalise = normalise, approx_sigmoid = "linear", linear_sig_coeff = 0.10), 
# run_sim(T, M = M, iters = iters, eta = eta, mode = "RL", normalise = normalise, approx_sigmoid = "linear", linear_sig_coeff = 0.207), 
# run_sim(T, M = M, iters = iters, eta = eta, mode = "RL_raw", normalise = normalise, approx_sigmoid = "linear"),]

# iters = int(0.5e4)
# datas = [run_sim(T, M = None, iters = iters, eta = eta, mode = "supervised", normalise = normalise, approx_sigmoid = "step"),
# run_sim(T, M = None, iters = iters, eta = eta, mode = "supervised", normalise = normalise, approx_sigmoid = "linear"),
# run_sim(T, M = 5, iters = iters, eta = eta, mode = "predictive", normalise = normalise, approx_sigmoid = "step"),]

iters = int(0.5e4)
datas = [
    run_sim(T, M = 4, iters = iters, eta = eta, mode = "predictive", normalise = True),
    run_sim(T, M = 4, iters = iters, eta = eta, mode = "predictive", normalise = False),
    run_sim(T, M = 4, iters = iters, eta = eta, mode = "predictive", normalise = False, orthogonal = True)
    ]

for i in range(len(titles)):
    plt.figure(figsize = (4,3))
    for data in datas:
        xs = np.linspace(0, 1, data.shape[0])
        plt.plot(xs, data[:, i])
    plt.title(titles[i])
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.xlim(0, 1)
    plt.show()

#%%

M = 4
T = 3
eta = 1e-2
iters = int(2e3)
mode = "predictive"
normalise = False

data1 = run_sim(T, M = M, iters = iters, eta = eta, mode = mode, normalise = normalise, orthogonal = True)
data2 = run_emp_sim(T, M = M, mode = mode, iters = iters, eta = eta, N = 1000, batch_size = 501, normalise = normalise, orthogonal = True)

#%%

for i in range(len(titles)):
    plt.figure(figsize = (4,3))
    for idata, data in enumerate([data2, data1]):
        xs = np.linspace(0, 1, data.shape[0])
        plt.plot(xs, data[:, i], ls = ["-", "--"][idata])
    plt.legend(["simulation", "theory"])
    plt.title(titles[i])
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.xlim(0, 1)
    plt.show()


#%% compare different learners

eta = 4e-3
iters = int(8e3)
T = 5
Ms = [1, 2, 4, 6, 20]
normalise = True

datas = [run_sim(T, M = None, iters = iters, eta = eta, mode = "supervised", normalise = normalise),
run_sim(T, M = None, iters = iters, eta = eta, mode = "RL", normalise = normalise, verbose = False),
run_sim(T, M = None, iters = iters, eta = eta, mode = "RL_raw", normalise = normalise, verbose = False)]
datas += [run_sim(T, M = M, iters = iters, eta = eta, mode = "predictive", normalise = normalise) for M in Ms]
datas = np.array(datas)

print(np.sum(datas[..., 0] < 0.9, axis = -1))
print(np.sum(datas[..., -1] < 0.9, axis = -1))

cols = [plt.get_cmap("tab10")(0), plt.get_cmap("tab10")(1), plt.get_cmap("tab10")(2)] + [np.zeros(3) + iM/(2+len(Ms)) for iM in range(len(Ms))]
labels = ["supervised", "RL (B = <R>)", "RL (B = 0)"] + [str(f"M = {M}") for M in Ms]

for i in range(len(titles)):
    plt.figure()
    for iD, data in enumerate(datas):
        plt.plot(data[:, i], color = cols[iD], label = labels[iD])
    plt.title(titles[i])
    plt.legend()
    plt.show()


#%% compare predictive and RL learners across different Ts


def plot_by_T(datas, prefix = ""):
    for i in range(len(titles)+1):
        plt.figure()
        for iT, T in enumerate(Ts):
            xs = datas[iT, :, -1]

            if i == len(titles):
                ys = np.log(datas[iT, :, 0])
            else:
                ys = datas[iT, :, i]
            plt.plot(xs, ys, color = np.zeros(3) + iT/(3+len(Ts)), label = f"T = {T}")
        
        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.title(prefix+"log rho" if i == len(titles) else titles[i])
        plt.legend()
        plt.show()

eta = 2e-2
base_iters = int(1.8e3)
M = 4
Ts = [2,4,6,8,10,12]
normalise = True
for mode in ["predictive", "RL"]:
    if mode == "predictive":
        iters = base_iters
    else:
        iters = 10*base_iters

    datas = []
    for T in Ts:
        datas.append(run_sim(T, M=M, iters = iters, eta = eta, normalise = normalise, mode = mode))
    plot_by_T(np.array(datas), prefix = f"{mode}: ")


#%% now run true 'simulated' learning

Ts = [2,4,6,8]

M = 4
eta = 2e-2
iters = int(9e3)
mode = "predictive"
normalise = True

data = []

for iT, T in enumerate(Ts):
    print("\n\n", T)
    data1 = run_sim(T, M = M, iters = iters, eta = eta, mode = mode, normalise = normalise)
    data2 = run_emp_sim(T, M = M, mode = mode, iters = iters, eta = eta, N = 1000, batch_size = 501, normalise = normalise, approx_sigmoid = "true", independent_samples = False)

    data.append([data2, data1])

#%%

datas = [np.array([dat[i] for dat in data]) for i in range(2)]

for i in range(len(titles)):
    plt.figure()
    for iT, T in enumerate(Ts):
        for idata in range(2):
            col = np.array([[0,1,0], [0,0,1]][idata]) * (iT+1)/(len(Ts)-0)
            label = (f"T = {T}" if idata == 0 else None)
            xs = datas[idata][iT, :, -1]
            plt.plot(xs, datas[idata][iT, :, i], color = col, label = label, ls = ["-", "--"][idata])
    plt.title(titles[i])
    plt.legend()
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.show()

#%% also try supervised learning on expected reward instead of log reward!

Ts = [2,4,6,8]
eta = 2e-2
iters = int(9e3)
mode = "supervised_nolog"
data = []

for iT, T in enumerate(Ts):
    print("\n\n", T)
    data.append(run_emp_sim(T, M = M, mode = mode, iters = iters, eta = eta, N = 1000, batch_size = 501, normalise = True))
plot_by_T(np.array(data))


#%% plot drho vs rho
T = 5
Ms = [2,4,10,20]
rhos = np.linspace(0.0, 1.0, 101)
all_rhos = []
all_rhos_emp = []
emp_inds = range(0, len(rhos), 10)

for irho, rho in enumerate(rhos):

    pg = calc_pg(rho)
    pR = calc_p_good_seq(pg, 1, T)# probability of sampling a correct sequence in one try

    sup_rho = beta*( 1 - rho**2  )

    angle = np.arccos(rho) # angle between vectors
    kappa = calc_kappa(rho)
    dw_teach = calc_rl_dw_teach(kappa, rho, pR, 0.25)
    RL_rho = dw_teach *(1 - rho**2 )

    M_rhos = []
    for M in Ms:
        pc = calc_pc(pg, M, T) # probability of training on a good action
        M_rhos.append(beta * (2*pc-1)*(1  - rho**2))


    all_rhos.append(np.array([sup_rho, RL_rho]+M_rhos))

    if irho in emp_inds:
        print(rho)
        M_rhos_emp = []
        for M in Ms:
            w, wstar = sample_w_wstar(251, rho)
            #grads, _ = estimate_grad(w, wstar, 40001, T, "step", None, "predictive", None, True, True) # batch, N, T
            grads, _ = estimate_grad("predictive", w, wstar, T, M = M, batch_size = 40001, approx_sigmoid = "step", independent_samples = True, orthogonal = False)
            wnew = w + 1e-5*grads.mean((0, 2))
            drho = np.sum(wnew * wstar)/np.sqrt(np.sum(wnew**2)) - np.sum(w*wstar)
            M_rhos_emp.append(1e5*drho)
        all_rhos_emp.append(np.array(M_rhos_emp))

all_rhos = np.array(all_rhos).T
all_rhos_emp = np.array(all_rhos_emp).T

#%%
for log in [False, True]:
    cols = [plt.get_cmap("tab10")(i) for i in [0,1]] + [np.zeros(3)+i/(len(Ms)+1) for i in range(len(Ms))]
    plt.figure()
    for i in range(len(all_rhos)):
        #ys = (all_rhos[i] / np.nanmax(all_rhos[i])) if norm else np.log(all_rhos[i])
        ys = np.log(all_rhos[i]) if log else all_rhos[i]
        plt.plot(rhos, ys, color = cols[i])
    for i in range(len(all_rhos_emp)):
        ys_emp = np.log(all_rhos_emp[i]) if log else all_rhos_emp[i]
        plt.scatter(rhos[emp_inds], ys_emp, marker = ".", s = 100, color = cols[i+2])
    plt.xlabel("rho")
    plt.ylabel("drho")
    #plt.xlim(0, 0.1)
    #plt.ylim(0.0075, 0.0095)
    #plt.ylim(0,0.04)
    plt.show()


#%% look at effect of temperature

Ms = [2,5, 10]
rhos = np.linspace(0.0, 1.0, 6)
scales = np.linspace(0.0, 3.0, 16)
N = 5001
data_sig = np.zeros((len(Ms), len(rhos), len(scales)))
data_ind = np.zeros((len(Ms), len(rhos)))
batch_size = 10001
T = 5

for iM, M in enumerate(Ms):
    print(M)
    for irho, rho in enumerate(rhos):
        
        w, wstar = sample_w_wstar(N, rho)

        X = np.random.normal(0, 1, (batch_size, N, T))
        Z = w @ X
        Ytrue = sign(wstar @ X) # batch x T
        
        pg = calc_pg(np.clip(np.sum(w*wstar)/np.sqrt(np.sum(w**2)), -1.0 + 1e-10, 1.0-1e-10))
        data_ind[iM, irho] = sample_independent_ytarget(Ytrue, np.clip(pg, 1e-10, 1.0-1e-10), M)[1]

        for iscale, scale in enumerate(scales):
            data_sig[iM, irho, iscale] = sample_sigmoid_ytarget(Ytrue, Z, scale, M)[1]

#%%
for iM in range(len(Ms)):
    plt.figure()
    for i in range(len(rhos)):
        col = np.zeros(3) + i/ (3+len(rhos))
        plt.plot(scales, data_sig[iM, i, :], color = col)
        plt.axhline(data_ind[iM, i], color = col, ls = "--")
    plt.xlabel("temperature")
    plt.ylabel("p(correct action)")
    plt.title(f"M = {Ms[iM]}")
    plt.show()


#%% verify signal-to-noise stuff

for mode in ["supervised", "predictive", "RL"]:

    M = 5
    Ts = np.arange(1,11)
    rhos = np.array([0.0, 0.3, 0.7])
    #mode = "supervised"
    data = np.zeros((len(rhos), len(Ts), 3)) # rho, sig, noise/sig
    emp_inds = range(0, len(Ts), 2)
    data_emp = np.zeros((len(rhos), len(emp_inds), 3))

    baseline = lambda pR: pR

    for irho, rho in enumerate(rhos):
        pg = calc_pg(rho)

        if mode == "RL":
            approx_sigmoid = "linear"
            pR = pg**Ts
            var = 0.113*pR*(1-pR)

            kappa = calc_kappa(rho)
            angle = np.arccos(rho)
            a1 = 0.5*kappa - beta
            a2 = 0.5*kappa - 0.25*np.sqrt(1 - rho**2)/(pi-angle)

            meansq = pR**2 * (a1*a1 + a2*a2 + rho*a1*a2)

        else:
            approx_sigmoid = "step"
            if mode == "supervised":
                pc = np.ones(len(Ts)) # always train on good action

            elif mode == "predictive":
                pc = calc_pc(pg, M, Ts) # train on sampled action

            var = 0.5 - 0.5*(2*pg - 1)*(2*pc - 1)
            meansq = beta**2 * ( 1 + (2*pc-1)**2 - 2*(2*pc-1)*rho )
        data[irho, ...] = np.array([meansq, var, np.log(var/meansq)]).T

        for iemp, iT in enumerate(emp_inds):
            T = Ts[iT]
            w, wstar = sample_w_wstar(501, rho)
            #grads, _ = estimate_grad(w, wstar, batch_size, T, approx_sigmoid, linear_sig_coeff, mode, baseline, independent_samples)
            grads, _ = estimate_grad("predictive", w, wstar, T, M = M, batch_size = 100001, approx_sigmoid = approx_sigmoid, linear_sig_coeff = 0.25, baseline = baseline, independent_samples = True, orthogonal = True)
            meansq_emp = N*(grads[..., 0].mean(0)**2).mean()
            var_emp = (grads**2).mean()
            data_emp[irho, iemp, :] = np.array([meansq_emp, var_emp, np.log(var_emp/meansq_emp)])
            print(rho, T, [np.round(a, 3) for a in [data[irho, iT, :], data_emp[irho, iemp, :]]])


    #%%

    fig, axs = plt.subplots(1, 3, figsize = (12,3.5))

    for ivals in range(3):
        for irho, rho in enumerate(rhos):
            col = plt.get_cmap("tab10")(irho) #np.zeros(3) + irho/(len(rhos)+1)
            #axs[ivals].plot(Ts, data[irho, :, ivals], color = col, label = f"rho = {rho}")
            axs[ivals].scatter(Ts[emp_inds], data_emp[irho, :, ivals], color = col, marker = ".", s = 300)
            
            axs[ivals].set_xlabel("T")
            axs[ivals].set_title(["N<w_i>^2", "<w_i^2>", "log(noise/sig)"][ivals])
            axs[ivals].set_xlim(Ts[0]-0.5, Ts[-1]+0.5)
        if ivals == 0:
            axs[ivals].legend()
    plt.show()

#%%



















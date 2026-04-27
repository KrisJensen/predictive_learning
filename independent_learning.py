
#%%
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import time
import os
import pickle

basedir = "/Users/kris/Documents/behrens/research/predictive_learning/"
print("loaded")

theta = lambda x: (x>0.0).astype(float)
sign = np.sign
pi = np.pi
sigma = lambda x: 1/(1+(np.exp((-x))))

# probability that a single action is good as a function of the student-teacher overlap
calc_pg = lambda rho: 0.5 + np.arcsin(rho) / np.pi

# probability that at least one sequence out of M is good
calc_p_good_seq = lambda pg, M, T: 1 - (1 - pg**T)**M

# probability that a single action is good when the entire sequence is not
calc_failed_pgood = lambda pg, T: pg * (1 - pg**(T-1)) / (1 - pg**T)

# probability that a single action is good when sampling M sequences and returning a correct one if it exists
def calc_pc(pg, M, T):
    p_good_seq = calc_p_good_seq(pg, M, T)
    return p_good_seq + (1-p_good_seq)*calc_failed_pgood(pg, T)

def calc_pc_emp(pg, M, T, K = 10000):
    corrects = np.random.binomial(1, p = pg, size = (K, M, T))
    num_correct = corrects.sum(-1)
    best_inds = np.argmax(num_correct, axis = -1)
    best_counts = num_correct[np.arange(K), best_inds]

    seqs = corrects[:, 0, :]
    all_correct = np.where(best_counts == T)[0]

    seqs[all_correct] = corrects[all_correct, best_inds[all_correct]]

    return seqs.mean(), len(all_correct)/K


beta = 1 / np.sqrt(2*np.pi)

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


# %% try to simulate a learning trajectory


def run_sim(T, M = 4, iters = int(1e5), eta = 1e-4, linear_sig_coeff = 0.25, mode = "predictive", normalise = False, verbose = False, approx_sigmoid = "step", rho = 0.0):

    assert approx_sigmoid in ["step", "linear"] # how do we approximate the sigmoid in our gradient
    assert mode in ["predictive", "supervised", "RL", "RL_raw"] # which learning algorithm
    #assert (not normalise) # doing it wrong

    if mode == "RL":
        baseline = lambda pR: pR
    if mode == "RL_raw":
        baseline = lambda pR: 0.0
        mode = "RL"
    if mode == "RL":
        approx_sigmoid = "linear"

    if approx_sigmoid == "linear":
        w_coeff = linear_sig_coeff
    elif approx_sigmoid == "step":
        w_coeff = beta

    sig_ss2 = 1.0
    sig_st2 = rho
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

        if mode == "predictive":
            pR = calc_p_good_seq(pg, M, T) # probability of sampling a correct sequence in M tries
            pc = calc_pc(pg, M, T) # probability of training on a good action
            dsig_st2 = eta*( beta*(2*pc - 1) - w_coeff*sig_st2/sig_ss)
            dsig_ss2 = 2*eta*( beta*(2*pc - 1)*sig_st2 - w_coeff*sig_ss)

        elif mode == "supervised":
            pR = calc_p_good_seq(pg, 1, T)# probability of sampling a correct sequence in one try
            pc = 1 # always training on a good action
            dsig_st2 = eta*( beta - w_coeff*sig_st2/sig_ss )
            dsig_ss2 = 2*eta*( beta*sig_st2 - w_coeff*sig_ss )

        elif mode == "RL":
            pR = calc_p_good_seq(pg, 1, T)# probability of sampling a correct sequence in one try
            pc = pg # probability of training on good action is simply probability of action being good
            angle = np.arccos(rho) # angle between vectors

            kappa = np.sqrt(pi) * np.sin((pi - angle)/2) / ((pi-angle)*np.sqrt(1+rho))

            B = baseline(pR)

            dw1 = 0.5*pR*kappa - w_coeff*pR + w_coeff*B - beta*B
            dw2 = 0.5*pR*kappa - w_coeff*pR*np.sin(angle)/(pi-angle)

            dsig_st2 = eta*( sig_st2/sig_ss*dw1 + dw2 )
            dsig_ss2 = 2*eta*( sig_ss*dw1 + sig_st2*dw2)


        # rhos, sigs_st, sigs_ss, pgs, pcs 
        if iter_ % int(np.floor(iters / 100)) == 0:
            data.append([rho, sig_st2, sig_ss, pg, pc, pR, iter_])

        if verbose and iter_ % 10000 == 0:
            print()
            print(data[-1])
            print(dsig_st2/eta, dsig_ss2/eta)

        # try:
        #     sig_ss2 = max(1e-10, sig_ss2+dsig_ss2)
        # except ValueError:
        #     print(sig_ss2, dsig_ss2.shape)
        #     print([np.shape(v) for v in [sig_ss, dw1, sig_st2, dw2, sig_st2, rho, pR, pc, kappa, w_coeff, angle, pi]])
        #     raise NotImplementedError

        sig_ss2 = max(1e-10, sig_ss2+dsig_ss2)
        sig_st2 += dsig_st2

    return np.array(data)

def sample_independent_ytarget(Ytrue, pg, M):
    batch_size, T = Ytrue.shape
    corrects = np.random.binomial(1, p = pg, size = (batch_size, M, T)) # decide which sequence elements are correct
    num_correct = corrects.sum(-1) # how many are correct in each sequence
    best_inds = np.argmax(num_correct, axis = -1) # what's the best one
    best_counts = num_correct[np.arange(batch_size), best_inds] # how many counts did it have

    seqs = corrects[:, 0, :] # pick our behaviour
    all_correct = np.where(best_counts == T)[0] # for the batches with a sequence that is entirely correct
    seqs[all_correct] = 1 # all good
    
    Ytarget = seqs * Ytrue - (1-seqs)*Ytrue
    pc = seqs.mean()

    return Ytarget, pc


def sample_sigmoid_ytarget(Ytrue, Z, scale, M):
    batch_size, T = Ytrue.shape

    sampler_pi = sigma(scale*Z)[:, None, :] * np.ones((batch_size, M, T))
    Ysamps = np.random.binomial(1, p = sampler_pi)*2.0-1.0
    corrects = (Ysamps == Ytrue[:, None, :]).astype(int)
    num_correct = corrects.sum(-1) # how many are correct in each sequence
    best_inds = np.argmax(num_correct, axis = -1) # what's the best one
    best_counts = num_correct[np.arange(batch_size), best_inds] # how many counts did it have

    Ytarget = Ysamps[:, 0, :] # pick our samples
    all_correct = np.where(best_counts == T)[0] # for the batches with a sequence that is entirely correct
    Ytarget[all_correct] = Ysamps[all_correct, best_inds[all_correct], :] # corresponding sequence
    
    pc = (Ytarget == Ytrue).mean()

    return Ytarget, pc




def run_emp_sim(T, M=4, N = 1000, iters = int(1e5), eta = 1e-4, batch_size = 1001, mode = "predictive", normalise = False, approx_sigmoid = "step", linear_sig_coeff = 0.25, overwrite = False, independent_samples = True, rho = 0, **kwargs):

    assert mode in ["predictive", "supervised", "RL", "RL_raw", "supervised_nolog"]
    assert approx_sigmoid in ["step", "linear", "true"]
    if mode != "predictive":
        M = None

    prms = [(k.item(), locals()[k]) for k in np.sort(list(locals().keys()))]
    prm_str = "_".join([p[0]+"-"+str(p[1]) for p in prms])
    print(prm_str)
    if (f"{prm_str}.p" in os.listdir(f"{basedir}/data/")) and (not overwrite):
        return pickle.load(open(f"{basedir}/data/{prm_str}.p", "rb"))

    if mode == "RL":
        baseline = lambda pR: pR
    if mode == "RL_raw":
        baseline = lambda pR: 0.0
        mode = "RL"

    if linear_sig_coeff == "beta":
        linear_sig_coeff = beta

    #raise NotImplementedError

    z0 = np.random.normal(0, 1, N)
    z1 = np.random.normal(0, 1, N)
    wstar = z0
    w = rho * z0 + np.sqrt(1-rho**2) * z1
    wstar = wstar / np.sqrt(np.square(wstar).sum())
    
    if rho == 0: # start from true zero
        w = w - np.sum(w*wstar)*wstar

    w = w / np.sqrt(np.square(w).sum())

    sig_tt, sig_tt2 = 1.0, 1.0
    t0 = time.time()
    data = []

    for iter_ in range(iters):

        # sample inputs
        X = np.random.normal(0, 1, (batch_size, N, T))
        Z = w @ X
        theta_Z = theta(Z) # batch x T
        Ytrue = sign(wstar @ X) # batch x T

        if normalise:
            w = w / np.sqrt(np.sum(np.square(w)))

        # compute correlations
        sig_ss2 = np.sum(w**2)
        sig_st2 = np.sum(w*wstar)
        sig_ss = np.sqrt(sig_ss2)
        rho = sig_st2/sig_ss
        pg = calc_pg(rho)
        pR = calc_p_good_seq(pg, (M if mode == "predictive" else 1), T)# probability of sampling a correct sequence

        # how do we approximate our sigmoid in the gradient update?
        if approx_sigmoid == "step":
            sig_Z = theta_Z # approximate sigmoid with step function
        elif approx_sigmoid == "true":
            sig_Z = sigma(Z)
        elif approx_sigmoid == "linear":
            sig_Z = 0.5 + linear_sig_coeff * Z / sig_ss # linearise sigmoid

        if mode == "predictive":
            if independent_samples: # cheat with sampling sequences
                Ytarget, pc = sample_independent_ytarget(Ytrue, pg, M)
            else: # sample from our sigmoids
                scale = 0.0+2*rho
                Ytarget, pc = sample_sigmoid_ytarget(Ytrue, Z, scale, M)
            G = 1
            dw = ((0.5 - sig_Z + 0.5*Ytarget)[:, None, :] * X).mean((0,2))

        elif mode == "supervised":
            pc = 1
            Ytarget = Ytrue
            G = 1
            dw = ((0.5 - sig_Z + 0.5*Ytrue)[:, None, :] * X).mean((0,2))

        elif mode == "RL":
            pc = pg
            Ytarget = theta_Z*2 - 1 # actions the agent took
            R = ((Ytarget == Ytrue).sum(-1) == T).astype(float) # reward if full sequence is good
            G = (R - baseline(pR))[:, None, None]

        if mode == "supervised_nolog":
            sigma_Z = sigma(Z)
            ptrue = 0.5 + Ytrue*(sigma_Z - 0.5)
            dptrue = Ytrue*sigma_Z*(1-sigma_Z)
            dw = np.zeros(N)
            pc = 1
            for t in range(T):
                not_t = np.arange(T)[np.arange(T) != t]
                dw += ((dptrue[:, t]*np.prod(ptrue[:, not_t], axis = -1))[:, None]*X[..., t]).mean(0)/T

        else:
            # update takes the same form in all cases
            dw = (G * (0.5 - sig_Z + 0.5*Ytarget)[:, None, :] * X).mean((0,2))

        w += eta*dw

        # rhos, sigs_st, sigs_ss, pgs, pcs 
        if iter_ % int(np.floor(iters / 100)) == 0:
            data.append([rho, sig_st2, sig_ss, pg, pc, pR, iter_])

        if iter_ % int(np.round(iters / 20)) == 0:
            print("\n", iter_, rho, np.round((time.time() - t0)/60, 1))
            print(np.sum(dw * wstar), beta*((2*pc - 1)*sig_tt - sig_st2/sig_ss))
            print(np.sum(dw * w), beta*((2*pc - 1)*sig_st2/sig_tt - sig_ss))

    pickle.dump(np.array(data), open(f"{basedir}/data/{prm_str}.p", "wb"))

    return np.array(data)

titles = ["student-teacher correlation", "student-teacher overlap", "student magnitude", "p(guess correct)", "p(action correct)", "p(sequence correct)"]

#%%

M = 4
T = 5
eta = 5e-3
normalise = True

iters = int(2e4)
datas = [run_sim(T, M = M, iters = iters, eta = eta, mode = "RL", normalise = normalise, approx_sigmoid = "linear", linear_sig_coeff = 0.25),
run_sim(T, M = M, iters = iters, eta = eta, mode = "RL", normalise = normalise, approx_sigmoid = "linear", linear_sig_coeff = 0.10), 
run_sim(T, M = M, iters = iters, eta = eta, mode = "RL", normalise = normalise, approx_sigmoid = "linear", linear_sig_coeff = 0.207), 
run_sim(T, M = M, iters = iters, eta = eta, mode = "RL_raw", normalise = normalise, approx_sigmoid = "linear"),]

iters = int(0.5e4)
datas = [run_sim(T, M = None, iters = iters, eta = eta, mode = "supervised", normalise = normalise, approx_sigmoid = "step"),
run_sim(T, M = None, iters = iters, eta = eta, mode = "supervised", normalise = normalise, approx_sigmoid = "linear"),
run_sim(T, M = 5, iters = iters, eta = eta, mode = "predictive", normalise = normalise, approx_sigmoid = "step"),]

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
iters = int(5e3)
mode = "predictive"
normalise = True

data1 = run_sim(T, M = M, iters = iters, eta = eta, mode = mode, normalise = normalise)
data2 = run_emp_sim(T, M = M, mode = mode, iters = iters, eta = eta, N = 1000, batch_size = 501, normalise = normalise)

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

eta = 4e-4
iters = int(8e4)
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


def plot_by_T(datas):
    for i in range(len(titles)):
        plt.figure()
        for iT, T in enumerate(Ts):
            xs = datas[iT, :, -1]
            plt.plot(xs, datas[iT, :, i], color = np.zeros(3) + iT/(3+len(Ts)), label = f"T = {T}")
        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.title(titles[i])
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
    plot_by_T(np.array(datas))


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
T = 10
Ms = [2,4,10,20]
rhos = np.linspace(0.0, 1.0, 101)
all_rhos = []
approx = True
for rho in rhos:
    sig_st2 = rho
    pg = calc_pg(rho)
    pR = calc_p_good_seq(pg, 1, T)# probability of sampling a correct sequence in one try

    sup_rho = beta*( 1 - rho**2  )

    angle = np.arccos(rho) # angle between vectors
    kappa = np.sqrt(pi) * np.sin((pi - angle)/2) / ((pi-angle)*np.sqrt(1+rho))
    dw1 = 0.5*pR*kappa - beta*pR
    dw2 = 0.5*pR*kappa - 0.25*pR*np.sqrt(1-rho**2)/(pi-angle)
    RL_rho = dw2 *(1 - rho**2 )

    M_rhos = []
    for M in Ms:
        pc = calc_pc(pg, M, T) # probability of training on a good action
        M_rhos.append(beta * (2*pc-1)*(1  - rho**2))

    all_rhos.append(np.array([sup_rho, RL_rho]+M_rhos))
    

all_rhos = np.array(all_rhos).T

for log in [False, True]:
    cols = [plt.get_cmap("tab10")(i) for i in [0,1]] + [np.zeros(3)+i/(len(Ms)+1) for i in range(len(Ms))]
    plt.figure()
    for i in range(len(all_rhos)):
        #ys = (all_rhos[i] / np.nanmax(all_rhos[i])) if norm else np.log(all_rhos[i])
        ys = np.log(all_rhos[i]) if log else all_rhos[i]
        plt.plot(rhos, ys, color = cols[i])
    plt.xlabel("rho")
    plt.ylabel("drho")
    plt.xlim(0, 0.1)
    #plt.ylim(0.0075, 0.0095)
    plt.ylim(0,0.04)
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
        z0 = np.random.normal(0, 1, N)
        z1 = np.random.normal(0, 1, N)
        wstar = z0
        w = rho * z0 + np.sqrt(1-rho**2) * z1
        wstar = wstar / np.sqrt(np.square(wstar).sum())
        w = w / np.sqrt(np.square(w).sum())

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

#%%

#%%


























# %% compute initial overlap vs T for different M

Ts = range(2, 20)
Ms = [1,3,1e1,1e2,1e3,1e4,1e5]
data = np.zeros((2, len(Ts), len(Ms)))
for irho, rho in enumerate([0.0, 0.2]):
    for iT, T in enumerate(Ts):
        for iM, M in enumerate(Ms):
            data[irho, iT, iM] = 2*calc_pc(calc_pg(rho), M, T)-1

plt.figure()
for irho in range(2):
    for iM, M in enumerate(Ms):
        plt.plot(data[irho, :, iM], color = plt.get_cmap("tab10")(iM), ls = ["-", "--"][irho])
plt.show()

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

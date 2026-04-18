
#%%
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import time

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


def run_sim(T, M = 4, iters = int(1e5), eta = 1e-4, mode = "predictive", normalise = False, verbose = False):

    assert mode in ["predictive", "supervised", "RL", "RL_raw"]

    sig_st2 = 0.0
    sig_ss2 = 1.0
    data = []

    for iter_ in range(iters):
        #rho = np.minimum(1-1e-10, sig_st2 / np.sqrt(sig_ss2))

        if normalise:
            sig_ss2 = 1

        sig_ss = np.sqrt(sig_ss2)

        rho = sig_st2 / sig_ss
        pg = calc_pg(rho)

        if mode == "predictive":
            pR = calc_p_good_seq(pg, M, T) # probability of sampling a correct sequence in M tries
            pc = calc_pc(pg, M, T) # probability of training on a good action
            dsig_st2 = eta*beta*(2*pc - 1 - sig_st2/sig_ss)
            dsig_ss2 = 2*eta*beta*( (2*pc - 1)*sig_st2 - sig_ss)

        elif mode == "supervised":
            pR = calc_p_good_seq(pg, 1, T)# probability of sampling a correct sequence in one try
            pc = 1 # always training on a good action
            dsig_st2 = eta*beta*(1 - sig_st2/sig_ss)
            dsig_ss2 = 2*eta*beta*( sig_st2 - sig_ss)

        elif mode == "RL":
            pR = calc_p_good_seq(pg, 1, T)# probability of sampling a correct sequence in one try
            pc = pg # probability of training on good action is simply probability of action being good
            p0star = calc_failed_pgood(pg, T) # probability that action is good given sequence is bad
            dsig_st2 = 2*eta*beta*pR*(1 + pR*p0star - pR - p0star)
            dsig_ss2 = 4*eta*beta*pR*(1 + pR*p0star - pR - p0star)*sig_st2

        elif mode == "RL_raw": # no baseline
            pR = calc_p_good_seq(pg, 1, T)# probability of sampling a correct sequence in one try
            pc = pg # probability of training on good action is simply probability of action being good
            dsig_st2 = eta*beta*pR*(1 - sig_st2/sig_ss)
            dsig_ss2 = 2*eta*beta*pR*( sig_st2 - sig_ss)

        # rhos, sigs_st, sigs_ss, pgs, pcs 
        if iter_ % 100 == 0:
            data.append([rho, sig_st2, sig_ss, pg, pc, pR])

        if verbose and iter_ % 10000 == 0:
            print()
            print(data[-1])
            print(dsig_st2/eta, dsig_ss2/eta)

        sig_st2 += dsig_st2
        sig_ss2 += dsig_ss2

    return np.array(data)

def sample_ytarget(Ytrue, pg, batch_size, M, T):
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

def run_emp_sim(T, M=4, N = 1000, iters = int(1e5), eta = 1e-4, batch_size = 1001, mode = "predictive", normalise = False):

    assert mode in ["predictive", "supervised", "RL", "RL_raw"]

    if mode == "RL":
        baseline = lambda pR: pR
    if mode == "RL_raw":
        baseline = lambda pR: 0.0
        mode = "RL"


    rho = 0.0

    z0 = np.random.normal(0, 1, N)
    z1 = np.random.normal(0, 1, N)
    wstar = z0
    w = rho * z0 + np.sqrt(1-rho**2) * z1
    wstar = wstar / np.sqrt(np.square(wstar).sum())
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

        # compute correlations
        sig_ss2 = np.sum(w**2)
        sig_st2 = np.sum(w*wstar)
        sig_ss = np.sqrt(sig_ss2)
        rho = sig_st2/sig_ss
        pg = calc_pg(rho)
        pR = calc_p_good_seq(pg, (M if mode == "predictive" else 1), T)# probability of sampling a correct sequence

        if mode == "predictive":
            # for now we cheat with sampling sequences
            Ytarget, pc = sample_ytarget(Ytrue, pg, batch_size, M, T)
            dw = ((0.5 - theta_Z + 0.5*Ytarget)[:, None, :] * X).mean((0,2))

        elif mode == "supervised":
            pc = 1
            dw = ((0.5 - theta_Z + 0.5*Ytrue)[:, None, :] * X).mean((0,2))

        elif mode == "RL":
            pc = pg
            Ystudent = theta_Z*2 - 1 # actions we take
            R = ((Ystudent == Ytrue).sum(-1) == T).astype(float) # reward if full sequence is good

            # compute RL loss
            #dw = ((R - baseline(pR))[:, None, None] * (0.5 - theta_Z + 0.5*Ystudent)[:, None, :] * X).mean((0,2))
            # dw = ((R - baseline(pR))[:, None, None] * (0.5 - sigma(Z) + 0.5*Ystudent)[:, None, :] * X).mean((0,2))
            dw = ((R - baseline(pR))[:, None, None] * (0.25*Z + 0.5*Ystudent)[:, None, :] * X).mean((0,2))

        w += eta*dw

        data.append([rho, sig_st2, sig_ss, pg, pc, pR])

        if iter_ % int(np.round(iters / 20)) == 0:
            print("\n", iter_, rho, np.round((time.time() - t0)/60, 1))
            print(np.sum(dw * wstar), beta*((2*pc - 1)*sig_tt - sig_st2/sig_ss))
            print(np.sum(dw * w), beta*((2*pc - 1)*sig_st2/sig_tt - sig_ss))


    return np.array(data)

titles = ["student-teacher correlation", "student-teacher overlap", "student magnitude", "p(guess correct)", "p(action correct)", "p(sequence correct)"]

#%%

M = 4
T = 3
eta = 25e-4
iters = int(1e4)
mode = "RL"

data1 = run_sim(T, M = M, iters = iters, eta = eta, mode = mode)
data2 = run_emp_sim(T, M = M, mode = mode, iters = iters, eta = eta, N = 1000, batch_size = 501)

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


#%% different learners

eta = 2e-4
iters = int(3.5e5)
T = 5
Ms = [2, 3, 4, 6, 10, 30]
normalise = True

datas = [run_sim(T, M = None, iters = iters, eta = eta, mode = "RL", normalise = normalise, verbose = False),
run_sim(T, M = None, iters = iters, eta = eta, mode = "RL_raw", normalise = normalise, verbose = False),]
datas += [run_sim(T, M = M, iters = iters, eta = eta, mode = "predictive", normalise = normalise) for M in Ms]
datas += [run_sim(T, M = None, iters = iters, eta = eta, mode = "supervised", normalise = normalise)]
datas = np.array(datas)

print(np.sum(datas[..., 0] < 0.9, axis = -1))
print(np.sum(datas[..., -1] < 0.9, axis = -1))

cols = [plt.get_cmap("tab10")(1), plt.get_cmap("tab10")(2)] + [np.zeros(3) + iM/(2+len(Ms)) for iM in range(len(Ms))]+ [plt.get_cmap("tab10")(0)]

for i in range(len(titles)):
    plt.figure()
    for iD, data in enumerate(datas):
        plt.plot(data[:, i], color = cols[iD])
    plt.title(titles[i])
    plt.show()


#%% predictive learning curve for different Ts

eta = 5e-4
iters = int(1e4)
M = 4
Ts = [2,3,4,5,6,7,8]

datas = []
for T in Ts:
    datas.append(run_sim(T, M=M, iters = iters, eta = eta))

datas = np.array(datas)

#%%

for i in range(len(titles)):
    plt.figure()
    for iT, T in enumerate(Ts):
        plt.plot(datas[iT, :, i], color = np.zeros(3) + iT/(3+len(Ts)))
    plt.title(titles[i])
    plt.show()



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

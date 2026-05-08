#%%

print(2+3)

#%%

import prsta
from prsta import basedir
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter1d
from torch import nn
from torch.nn.functional import one_hot
import pickle
prsta.reload()
np.random.seed(1)
torch.manual_seed(1)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#%%
prsta.reload()

#%%

# instantiate model
Nrec = 700
max_steps = 6
side_length = 4
Nloc = side_length**2
env = prsta.envs.MazeEnv(batch_size = 201, max_steps = max_steps, side_length = side_length)
rnn = prsta.agents.VanillaRNN(env, Nrec = Nrec).to(device)
rnn.store_all_activity = False # make sure to store all activity for later analysis


obs = env.observation()

loss = rnn.forward()

acc = (rnn.action == rnn.optimal_actions.argmax(-1)).to(float).mean().item()

print(acc)

#%% instantiate optimizer and some variables to keep track of
optim = torch.optim.Adam(rnn.parameters(), lr=3e-4)
all_epochs, all_losses, all_accs = [], [], []

# now run actual training loop
num_epochs = 20000
eval_freq = 25
t0 = time.time()
for epoch in range(num_epochs):

    # reset env
    with torch.no_grad():
        env.reset()
        
    # run all the steps
    for t in range(env.max_steps):
        optim.zero_grad() # reset gradient accumulator
        loss = rnn.forward(reset_env = False) # run RNN and compute loss
        loss.backward() # compute gradients
        optim.step() # update parameters
        
        with torch.no_grad(): # update env
            env.step(rnn.action.cpu())
    
    with torch.no_grad():
        if epoch % eval_freq == 0:
            loss = rnn.forward().item()
            acc = (rnn.action == rnn.optimal_actions.argmax(-1)).to(float).mean().item()
            all_epochs.append(epoch)
            all_losses.append(loss)
            all_accs.append(acc)
            
            losses = [np.round(l.item(), 4) for l in [rnn.acc_loss, rnn.ent_loss, rnn.weight_loss, rnn.rate_loss]]
            print(epoch, loss, acc, np.round((time.time() - t0)/60, 2), losses)


data = {"env": env, "rnn": rnn.cpu(), "all_epochs": all_epochs, "all_losses": all_losses, "all_accs": all_accs}
pickle.dump(data, open(f"{basedir}/models/test.p", "wb"))


#%%

data = pickle.load(open(f"{basedir}/models/test.p", "rb"))
rnn = data["rnn"].to(device)
env = data["env"]
max_steps = env.max_steps
Nloc = env.num_locs
batch_inds = env.batch_inds
all_epochs = data["all_epochs"]
all_losses = data["all_losses"]
all_accs = data["all_accs"]

smooth_loss = gaussian_filter1d(all_losses, sigma = 3)
smooth_acc = gaussian_filter1d(all_accs, sigma = 3)

plt.figure()
fig, axs = plt.subplots(1, 2, figsize = (8, 3))
axs[0].plot(smooth_loss, label = "smooth loss")
axs[1].plot(smooth_acc, alpha = 0.5, label = "smooth acc")
plt.show()

#%% now try to run a 'hippocampal optimisation'

def get_best_path(env):
    
    with torch.no_grad():
        
        all_locs = torch.zeros(env.batch, env.max_steps+1, dtype = int)
        all_locs[:, 0] = env.loc
        for t in range(env.max_steps):
            loc = all_locs[:, t]
            # we need to find the actions with optimal long-term value
            new_locs = env.neighbors[env.batch_inds, loc, :] # for each action, where do I end up
            
            vnext = torch.zeros(env.batch, env.output_dim) - torch.inf # value of each action (default to -inf for impossible actions)
            for a in range(env.num_actions): # for each action
                new_locs_a = new_locs[:, a] # where would I end up
                vnext[env.batch_inds, new_locs_a] = env.vs[env.batch_inds, t+1, new_locs_a] # what would the value be here

            all_locs[:, t+1] = torch.argmax(vnext, -1) # which action has the highest value?
    
    emp_rews = torch.stack([env.rews[env.batch_inds, t, all_locs[:, t]] for t in range(env.max_steps+1)])
    assert torch.allclose(emp_rews.sum(0), env.vs[env.batch_inds, 0, env.loc])
    
    return all_locs # batch, steps+1

def calc_overlap(Cs):
    # first normalize coefficients across neurons because we want the angles not the magnitudes
    norm_Cs = Cs / (Cs**2).sum(-1, keepdims = True).sqrt()
    # now compute the overlap for every pair of slots and locations
    overlap =  (norm_Cs[:, None, :, None, ...] * norm_Cs[None, :, None, ...]).sum(-1) # (num_slots, num_slots, locs, locs)
    return overlap

def pred_logprobs(rs, Cs, bs):
    logps = (Cs @ rs[:, None, :, None])[..., 0] + bs[None, ...]
    logps = logps - logps.logsumexp(-1, keepdim = True)
    return logps

def pred_loss(rs, locs, Cs, bs, L2_reg = 1e-5, olap_reg = 0.0):
    
    logps = pred_logprobs(rs, Cs, bs)
    loss = (-logps*locs).sum((1,2)).mean() # sum across time and locs, mean across batch
    loss = loss + L2_reg*( (Cs**2).mean() + (bs**2).mean() ) # cross-entropy loss + regularisation
    
    if olap_reg > 1e-10:
        loss = loss + olap_reg * calc_overlap(Cs).abs().sum() # overlap regularisation

    return loss

#%% collect a bunch of trial data
all_rs = []
all_best_paths = []
all_real_paths = []

num_epochs = 200

rnn.greedy = False
with torch.no_grad():
    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            print(epoch)
        
        env.reset()
        new_real_path = torch.zeros(env.batch, env.max_steps+1, dtype = int) + env.loc[:, None] # initial location
        for t in range(max_steps):
            rnn.forward(reset_env = False) # run neural dynamics
            if t == 0: # after first step, store initial neural activity and optimal path
                all_rs.append(rnn.r.detach().cpu()) # store initial activity
                all_best_paths.append(get_best_path(env).detach().cpu()) # store future optimal path
            
            env.step(rnn.action) # update location
            new_real_path[:, t+1] = env.loc # store the new location of the agent
        
        all_real_paths.append(new_real_path.detach().cpu()) # store the real path taken by the agent
        
    all_rs, all_best_paths, all_real_paths = torch.cat(all_rs, 0)[..., 0], torch.cat(all_best_paths, 0), torch.cat(all_real_paths, 0)

print((all_best_paths == all_real_paths).to(float).mean(0))

#%% can decode future better than optimal _when greedy_

train_x, test_x = all_rs[0::2], all_rs[1::2]
#train_y, test_y = [one_hot(arr, Nloc).to(torch.float32) for arr in [all_best_paths[0::2, 1:], all_best_paths[1::2, 1:]]]
train_y, test_y = [one_hot(arr, Nloc).to(torch.float32) for arr in [all_real_paths[0::2, 1:], all_real_paths[1::2, 1:]]]
    
Cs = nn.Parameter(torch.zeros(max_steps, Nloc, Nrec))
bs = nn.Parameter(torch.zeros(max_steps, Nloc))


optim2 = torch.optim.Adam([Cs, bs], lr = 1e-2)

epochs = 200
for epoch in range(epochs):
    optim2.zero_grad()
    loss = pred_loss(train_x, train_y, Cs, bs, L2_reg = 1e-5, olap_reg = (0.0 if epoch < 60 else 5e-3))
    loss.backward()
    optim2.step()
    
    with torch.no_grad():
        if epoch % 10 == 0:
            test_ps = pred_logprobs(test_x, Cs, bs)
            test_acc = (test_ps.argmax(-1) == test_y.argmax(-1)).to(float).mean(0).detach().cpu().numpy()
            print(epoch, loss.item(), test_acc)

Cs, bs = Cs.detach().cpu(), bs.detach().cpu()
Csnorm = Cs / torch.square(Cs).sum(-1, keepdims = True).sqrt()
olap = (Csnorm[None, ...] * Csnorm[:, None, :]).sum(-1).abs().mean(-1) # average overlap across locs for each pair of subspaces

print(torch.round(olap, decimals = 2))

#pickle.dump([Cs, bs], open(f"{basedir}/models/decoders.p", "wb"))

#%%
#Cs, bs = pickle.load(open(f"{basedir}/models/decoders.p", "rb"))

            
#%% now try to run a 'hippocampal simulation' and see if we can increase performance!
seed = 0
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

def sample_seq(loc, pis, adjacency, greedy = False):
    curr_loc = loc.clone() # batch
    batch_inds = torch.arange(loc.shape[0])
    pi_as = torch.zeros(loc.shape[0], max_steps)
    locs = torch.zeros(loc.shape[0], max_steps, dtype = int)
    for t in range(max_steps):
        adj = adjacency[batch_inds, curr_loc, :] # renormalize over adjacent states (batch, Nloc)
        pi = (pis[:, t, :]+1e-20) * adj # add some jitter to make sure the policy is not exactly zero. Otherwise we sometimes run into nans
        pi = pi/pi.sum(-1, keepdims = True) # normalise (batch, Nloc)
        
        if greedy:
            curr_loc = torch.argmax(pi, -1)
        else:
            curr_loc = torch.multinomial(pi, 1)[..., 0]
            
        locs[:, t] = curr_loc
        pi_as[:, t] = pi[batch_inds, curr_loc] # what was the probability of going here?
    
    return locs, pi_as

def sequence_rew(rews, path):
    batch_inds = torch.arange(rews.shape[0])
    # simply the reward available at each location
    step_rews = torch.stack([rews[batch_inds, t, path[:, t]] for t in range(path.shape[1])])
    return step_rews.T

def eval(pis, loc, rews, adjacency, iters = 1000, greedy = False):
    if greedy:
        iters = 1
    Rs = torch.zeros(iters, pis.shape[0], pis.shape[1]) # iters by batch by step
    for iter_ in range(iters):
        samp_seq = sample_seq(loc, pis, adjacency, greedy = greedy)[0]
        Rs[iter_, ...]= sequence_rew(rews, samp_seq)
    return Rs.mean(0) # batch, step

def optimise_representation(rs, baseline = torch.tensor(0.0), lr = 0.2, plan_batch = 1, num_batches = 40, print_every = 1e6, compare_greedy = True):

    for iter_ in range(num_batches):
        
        grads = torch.zeros(plan_batch, rs.shape[0], Nrec)
        
        for p in range(plan_batch):
            logpis = pred_logprobs(rs, Cs, bs) # this our current policy
            pis = logpis.exp()
            
            samp_seq, action_probs = sample_seq(loc, pis, adjacency, greedy = False)
            R_seq = sequence_rew(rews, samp_seq) # how good is it?(batch, step)
            
            if len(baseline.shape) == 2: # for each timepoint separately
                R_togo = R_seq.flip(-1).cumsum(-1).flip(-1) #(batch, steps)
                td_err = R_togo - baseline
            else:
                td_err = (R_seq.sum(-1) - baseline)[:, None] # what is the TD error (batch, 1) 
            
            if (iter_ % print_every == 0) and (p == 0):
                
                if compare_greedy:
                    greedy_R = eval(pis, loc, rews, adjacency, greedy = True) # greedy reward (batch, step)
                
                print(iter_, greedy_R.sum(-1).mean(), (td_err[:, 0] > 0.0).to(torch.float).mean(), pis.amax(-1).median(0).values)

            samp_1hot = one_hot(samp_seq, Nloc) # one-hot version (batch, step, Nloc)
            action_prms = (samp_1hot[..., None]*Cs[None, ...]).sum(2) # parameters associated with the action I took for each trial and time (batch, step, Nrec)

            grad = ( action_prms*(td_err*(1.0 - action_probs))[..., None] ).sum(1) # policy gradient
            grads[p, ...] = grad
            
        rs = rs + lr*grad.mean(0) # update my firing rates
    
    return rs


env.reset()
loc, rews, adjacency = env.loc.detach().clone(), env.rews.detach().clone(), env.adjacency.detach().clone()
rnn.forward(reset_env = False) # run some random trial

# this is our baseline activity that we want to improve
rs0 = rnn.r.detach().cpu()[..., 0]
logpis0 = pred_logprobs(rs0, Cs, bs) # this our base policy
pis0 = logpis0.exp()
Rg0 = eval(pis0, loc, rews, adjacency, greedy = True)
R0 = eval(pis0, loc, rews, adjacency, iters = 2000)

rs = rnn.r.detach().cpu()[..., 0]

Vt = Rg0.flip(-1).cumsum(-1).flip(-1)
Vtot = Vt[:, 0]

rs = optimise_representation(rs, baseline = Vtot, lr = 0.2, plan_batch = 1, num_batches = 40, print_every = 1e6, compare_greedy = True)

logpis1 = pred_logprobs(rs, Cs, bs) # this our final policy
pis1 = logpis1.exp()
R1 = eval(pis1, loc, rews, adjacency, iters = 2000) # how good is it

# how does the expected reward change

diffs = (R1 - R0).sum(-1)
maxdiff = np.abs(diffs).max()
bins = np.linspace(-maxdiff, maxdiff, 21)
plt.figure()
plt.hist(diffs, bins = bins)
plt.axvline(diffs.mean(), color = "k", zorder = 10)
plt.xlim(-maxdiff, maxdiff)
plt.show()

#% how does the greedy policy change
Rg1 = eval(pis1, loc, rews, adjacency, greedy = True)
print(diffs.mean(), diffs.std()/np.sqrt(len(diffs)))
print((Rg1-Rg0).sum(-1).mean(), (Rg1-Rg0).sum(-1).std()/np.sqrt(len(Rg0)))

#% how does the optimal first action change

popt0 = (rnn.optimal_actions * pis0[:, 0, :]).sum(-1)
popt1 = (rnn.optimal_actions * pis1[:, 0, :]).sum(-1)

print(popt0.mean(), popt1.mean())

# %%

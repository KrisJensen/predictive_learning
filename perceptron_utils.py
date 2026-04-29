
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import time
import os
import pickle

basedir = "/Users/kris/Documents/behrens/research/predictive_learning/"
print("loaded")

# first define some helper functions

theta = lambda x: (x>0.0).astype(float)
sign = np.sign
pi = np.pi
sigma = lambda x: 1/(1+(np.exp((-x))))

# sampling student and teacher
def sample_w_wstar(N, rho):
    wstar = np.random.normal(0, 1, N)
    w = rho * wstar + np.sqrt(1-rho**2) * np.random.normal(0, 1, N)
    wstar = wstar / np.sqrt(np.square(wstar).sum())
    
    if rho == 0: # start from true zero
        w = w - np.sum(w*wstar)*wstar
    w = w / np.sqrt(np.square(w).sum())

    return w, wstar

# probability that a single action is good as a function of the student-teacher overlap
calc_pg = lambda rho: 0.5 + np.arcsin(rho) / np.pi

# probability that at least one sequence out of M is good
calc_p_good_seq = lambda pg, M, T: 1 - (1 - pg**T)**M

# probability that a single action is good when the entire sequence is not
calc_failed_pgood = lambda pg, T: pg * (1 - pg**(T-1)) / (1 - pg**T)

# compute 'kappa'
def calc_kappa(rho):
    angle = np.arccos(rho) # angle between vectors
    kappa = np.sqrt(pi) * np.sin((pi - angle)/2) / ((pi-angle)*np.sqrt(1+rho))
    return kappa

def calc_rl_dw_stud(kappa, pR, w_coeff, B, beta):
    return 0.5*pR*kappa - w_coeff*pR + w_coeff*B - beta*B

def calc_rl_dw_teach(kappa, rho, pR, w_coeff):
    angle = np.arccos(rho) # angle between vectors
    return 0.5*pR*kappa - w_coeff*pR*np.sin(angle)/(pi-angle)

# probability that a single action is good when sampling M sequences and returning a correct one if it exists
def calc_pc(pg, M, T):
    p_good_seq = calc_p_good_seq(pg, M, T)
    return p_good_seq + (1-p_good_seq)*calc_failed_pgood(pg, T)

# function for validating empirically
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


#%% some functions for sampling actions in a 'planner'


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



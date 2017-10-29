# -------------------------------------------------------------------------
# 
#    Hyperparameter turning. Requires BayesOpt.
#
#    Copyright (C) 2017 Pooya Ronagh
# 
# ------------------------------------------------------------------------

import bayesopt
from bayesoptmodule import BayesOptContinuous, BayesOptDiscrete
import numpy as np
import itertools
from time import clock
import sys

class BayesOptTest(BayesOptContinuous):
    
    def evaluateSample(self, x):
	    total = 5.0
	    index= 1
	    for value in x:
	        total = total + ((value) - (index + 0.33))**2
	        index += 1
	    return total

hyperparam = {}
hyperparam['n_iterations'] = 50
hyperparam['n_iter_relearn'] = 5
hyperparam['n_init_samples'] = 2
hyperparam['noise']= 1e-10
# hyperparam['l_type']= 'L_MCMC'
hyperparam['kernel_name'] = "kMaternARD5"
hyperparam['kernel_hp_mean'] = [1]
hyperparam['kernel_hp_std'] = [5]
hyperparam['surr_name'] = "sStudentTProcessNIG"
#hyperparam['crit_name'] = "cMI"

N = 5
lb = np.zeros((N,))
ub = 10*np.ones((N,))

engine = BayesOptTest(N)
engine.parameters = hyperparam
engine.lower_bound = lb
engine.upper_bound = ub

engine.x_set = np.asarray([elt for elt in itertools.product(\
	range(10), repeat= N)], dtype= np.double)

start = clock()
mvalue, x_out, error = engine.optimize()

print("Result", mvalue, "at", x_out)
print("Running time:", clock() - start, "seconds")

# value = np.array([engine.evaluateSample(i) for i in engine.x_set])
# print("Optimum", value.min(), "at", engine.x_set[value.argmin()])


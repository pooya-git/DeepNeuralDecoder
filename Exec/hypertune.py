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
import json
import os

sys.path.insert(0, 'Trainer')
from ExRecCNOTFullLSTM import train
from ExRecCNOTFullLSTM import get_data


class BayesOptTest(BayesOptContinuous):

	def __init__(self, N, filename, param,\
			raw_data, p, lu_avg, lu_std, data_size):
		super(BayesOptTest, self).__init__(N)
		self.filename= filename
		self.param= param
		self.raw_data= raw_data
		self.p= p
		self.lu_avg= lu_avg
		self.lu_std= lu_std
		self.data_size= data_size

	def evaluateSample(self, x):

		param['opt']['batch size']= int(x[0] * 100) 
		param['opt']['learning rate']= 10**x[1] 
		param['opt']['iterations']= int(x[2] * 10)
		param['opt']['momentum']= x[3] 
		param['opt']['decay']= x[4]
		param['nn']['num hidden']= int(x[5] * 10)
		param['nn']['W std']= 10**x[6]
		param['nn']['b std']= 10**x[7]
		output= train(self.filename, self.param,\
			self.raw_data, self.p, self.lu_avg, self.lu_std, self.data_size)
		nn_avg= output['res']['nn avg']
		return nn_avg

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

if __name__ == '__main__':

    with open(sys.argv[1]) as paramfile:
        param = json.load(paramfile)

    output= []
    datafolder= sys.argv[2]

    for filename in os.listdir(datafolder):

		print("Reading data from " + filename)
		raw_data, p, lu_avg, lu_std, data_size = get_data(datafolder + filename)

		N = 8
		lb = np.array([1., -6., 1., .8, .8, 5., -2., -2.])
		ub = np.array([10., -2., 10., .99, .99, 100., 1., 1.])

		engine = BayesOptTest(N, datafolder + filename, param,\
			raw_data, p, lu_avg, lu_std, data_size)
		engine.parameters = hyperparam
		engine.lower_bound = lb
		engine.upper_bound = ub

		start = clock()
		mvalue, x_out, error = engine.optimize()

		print("Result", mvalue, "at", x_out)
		print("Running time:", clock() - start, "seconds")


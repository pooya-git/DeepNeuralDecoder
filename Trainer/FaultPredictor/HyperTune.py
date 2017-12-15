from bayesoptmodule import BayesOptContinuous, BayesOptDiscrete
from util import raise_ten, int_times_ten, identity, \
    activation_category, boolean_category
import numpy as np
import traceback
import logging
from pprint import pprint
from time import time
import json
from copy import deepcopy

class Domain():

    def __init__(self, vars, depth):

        self.lb= []
        self.ub= []
        self.loc= []
        self.func= []
        for key in vars.keys():
            if (key.split('.')[-1]=='num hidden'):
                num_rounds= depth
            elif (key.split('.')[-1]=='activations'):
                num_rounds= depth          
            else:
                num_rounds= 1
            for i in range(num_rounds):
                self.loc.append(key.split('.'))
                self.lb.append(vars[key][0])
                self.ub.append(vars[key][1])
                if (vars[key][2]=='int_times_ten'):
                    self.func.append(int_times_ten)
                elif (vars[key][2]=='raise_ten'):
                    self.func.append(raise_ten)
                elif (vars[key][2]=='identity'):
                    self.func.append(identity)
                elif (vars[key][2]=='activation_category'):
                    self.func.append(activation_category)
                elif (vars[key][2]=='boolean_category'):
                    self.func.append(boolean_category)
                else:
                    raise Exception('Function not recognized.')                    
        self.num_vars= len(self.lb)
        assert(self.num_vars==len(self.ub))
        assert(self.num_vars==len(self.loc))
        assert(self.num_vars==len(self.func))

class BayesOptTest(BayesOptContinuous):

    def __init__(self, m, param, hyperparam):

        self.m = m
        self.param= param
        self.inner_iter= hyperparam['env']['inner iterations']
        self.count= 1
        self.best_solution= None
        self.best_sample= None
        self.best_param= None

        test_fraction= self.param['data']['test fraction']
        tune_size= int(hyperparam['env']['database usage'] * self.m.data_size)
        self.m.test_size= int(test_fraction * tune_size)
        self.m.train_size= tune_size - self.m.test_size
        self.depth= len(param['nn']['num hidden'])
        self.domain= Domain(hyperparam['vars'], self.depth)

        super(BayesOptTest, self).__init__(self.domain.num_vars)
        self.parameters = hyperparam['bayesopt']
        self.lower_bound = np.array(self.domain.lb)
        self.upper_bound = np.array(self.domain.ub)
        
    def evaluateSample(self, x):
        
        print('## Iteration: '+ str(self.count))
        print('## New query: '+ ' '.join(str(elt) for elt in x))
        print('## Items: '+ ' '.join('.'.join(str(s) for s in elt) \
                                     for elt in self.domain.loc))
        self.count+=1

        depth_counter= 0
        activation_counter= 0
        for loc, func, val in zip(self.domain.loc, self.domain.func, x):
            if (loc[-1]=='num hidden'):
                leaf= reduce(dict.get, loc[:-1], self.param)
                leaf[loc[-1]][depth_counter]= func(val)
                depth_counter+=1
            elif (loc[-1]=='activations'):
                leaf= reduce(dict.get, loc[:-1], self.param)
                leaf[loc[-1]][activation_counter]= func(val)
                activation_counter+=1                
            else:
                leaf= reduce(dict.get, loc[:-1], self.param)
                leaf[loc[-1]]= func(val)

        print(json.dumps({'opt': self.param['opt'], \
                          'nn': self.param['nn']}, indent=2))
        batch_size= self.param['opt']['batch size']
        self.m.num_batches = self.m.train_size // batch_size
        self.m.error_scale= 1.0 * self.m.data_size / self.m.total_size

        fault_rates= []
        for i in range(self.inner_iter):
            try:
                if (self.param['nn']['iso']):
                    prediction, test_beg= self.m.iso_train(self.param)
                elif (self.param['nn']['mixed']):                
                    prediction, test_beg= self.m.mixed_train(self.param)
                else:
                    prediction, test_beg= self.m.train(self.param)
                print('Testing ...'),
                start_time= time()
                result= self.m.error_scale * \
                    self.m.num_logical_fault(prediction, test_beg)
                print('Done in ' + '{0:.2f}'.format(time() - start_time) + 's.')
                print('# Result: ' + str(result))
                fault_rates.append(result)
            except Exception as e:
                logging.error(traceback.format_exc())
                if not fault_rates:
                    if (self.best_solution == None):
                        return 1.0
                    else:
                        return 10.0 * self.best_solution
                else:
                    break

        this_solution= np.mean(fault_rates)
        print('## This value: ' + str(this_solution))

        if (self.best_solution == None or this_solution < self.best_solution):
            print('** New best **')
            self.best_solution = this_solution
            self.best_sample= x
            self.best_param= deepcopy(self.param)
            self.best_param['opt']['tune value']= this_solution
        else:
            print('### Best observed value: ' + str(self.best_solution))
            print('### Best observed query: ' + \
                  ' '.join(str(elt) for elt in self.best_sample))

        return this_solution

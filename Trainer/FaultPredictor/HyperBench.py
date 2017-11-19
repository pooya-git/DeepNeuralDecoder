from builtins import range
import bayesopt, os, sys
from bayesoptmodule import BayesOptContinuous, BayesOptDiscrete
import ExRecCNOTLSTM as decoder
import numpy as np
import cPickle as pickle
from ExRecCNOTData import *
from time import localtime, strftime, clock

class BayesOptTest(BayesOptContinuous):

    def __init__(self, m, hyper_param, lb, ub, keys):
        super(BayesOptTest, self).__init__(len(lb))
        self.parameters = hyper_param
        self.lower_bound = lb
        self.upper_bound = ub
        
        self.m = m
        self.keys= keys
        self.best_solution= None        
        self.num_classes= 2
        self.num_inputs= 2
        self.input_size= 6

        self.best_log= {}
        self.best_log['data']= {}
        self.best_log['opt']= {}
        self.best_log['res']= {}
        self.best_log['res']['p']= m.p
        self.best_log['res']['lu avg']= m.lu_avg
        self.best_log['res']['lu std']= m.lu_std

    def evaluateSample(self, x):
        
        print('# New sample='+ str(x))
        for elt, key in zip(x, keys):
            self.m.param[key[0]][key[1]]= key[2](elt)
        batch_size= self.m.param['opt']['batch size']
        test_fraction= self.m.param['data']['test fraction']
        num_trials= self.m.param['data']['num trials']
        self.m.test_size= int(test_fraction * self.m.data_size)
        self.m.train_size= self.m.data_size - self.m.test_size
        self.m.num_batches = self.m.train_size // batch_size
        self.m.error_scale= 1.0 * self.m.data_size / self.m.total_size

        fault_rates= []
        for i in range(num_trials):
            fault_rates.append(\
                train(m, self.num_classes, self.num_inputs, self.input_size, i))

        solution= np.mean(fault_rates)
        if (self.best_solution == None or solution < self.best_solution):
            print('* New best= ' + str(solution))
            self.best_solution = np.mean(fault_rates)
            self.best_log['param']= self.m.param
            self.best_log['data']['fault scale']= self.m.error_scale
            self.best_log['data']['total size']= self.m.total_size
            self.best_log['data']['test size']= self.m.test_size
            self.best_log['data']['train size']= self.m.train_size
            self.best_log['opt']['batch size']= batch_size
            self.best_log['opt']['number of batches']= self.m.num_batches
            self.best_log['res']['nn res'] = fault_rates
            self.best_log['res']['nn avg'] = solution
            self.best_log['res']['nn std'] = np.std(fault_rates)
        else:
            print('# Result= '+ str(solution))
        return solution
    
def raise_ten(elt):
    return 10**elt

def int_times_ten(elt):
    return int(10 * elt)

def identity(elt):
    return elt

if __name__ == '__main__':

    init_param= {}
    init_param['nn']= {}
    init_param['opt']= {}
    init_param['data']= {}
    init_param['usr']= {}
    init_param['nn']['num hidden']= 500
    init_param['nn']['W std']= 0.01
    init_param['nn']['b std']= 0.00
    init_param['opt']['batch size']= 1000
    init_param['opt']['learning rate']= 10.0**(-5)
    init_param['opt']['iterations']= 20
    init_param['opt']['momentum']= 0.99
    init_param['opt']['decay']= 0.98
    init_param['data']['test fraction']= 0.1
    init_param['data']['num trials']= 10
    init_param['usr']['verbose']= True
    init_param['nn']['type']= 'ExRecCNOTLabLSTM'

    hyper_param = {}
    hyper_param['n_iterations'] = 10
    hyper_param['n_iter_relearn'] = 5
    hyper_param['n_init_samples'] = 2
    hyper_param['noise']= 1e-10
    hyper_param['kernel_name'] = "kMaternARD5"
    hyper_param['kernel_hp_mean'] = [1]
    hyper_param['kernel_hp_std'] = [5]
    hyper_param['surr_name'] = "sStudentTProcessNIG"
    # hyper_param['l_type']= 'L_MCMC'
    # hyper_param['crit_name'] = "cMI"

    lb = np.array([-7,   50, -5.,  50,  0.85, 0.85])
    ub = np.array([-3,  100,  0., 100,  0.99, 0.99])
    keys = [('opt', 'learning rate', raise_ten), \
            ('nn', 'num hidden', int_times_ten), \
            ('nn', 'W std', raise_ten), \
            ('opt', 'batch size', int_times_ten), \
            ('opt', 'momentum', identity), \
            ('opt', 'decay', identity)]

    datafolder= '../../Data/SteaneCNOT_Pkl/e-04/'
    file_list= os.listdir(datafolder)
    outstream= {}

    count= 0
    for filename in file_list:
        if count>5: break
        else: count+=1

        with open(datafolder + filename, 'rb') as input_file:
            print('## Pickling model from ' + filename)
            m = pickle.load(input_file)

        m.param = init_param
        engine = BayesOptTest(m, hyper_param, lb, ub, keys)
        start = clock()
        mvalue, x_out, error = engine.optimize()
        outstream[m.p]= engine.best_log
        print('## Result' + str(mvalue) + 'at' + str(x_out))
        print('## Running time:' + str(clock() - start) + 'seconds')

    outfilename = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    f = open('../../Reports/SteaneCNOT/' + outfilename + '.json', 'w')
    f.write(json.dumps(outstream, indent=2))
    f.close()

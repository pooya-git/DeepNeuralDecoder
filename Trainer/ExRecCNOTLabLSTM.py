# ------------------------------------------------------------------------------
# 
#    CNOTExRec trainer. Use 4 RNNs with 1 LSTM cell to train X & Z at same time.
#
#    Copyright (C) 2017 Pooya Ronagh
# 
# ------------------------------------------------------------------------------

from builtins import range
import numpy as np
import tensorflow as tf
import sys
from util import y2indicator
import threading
import sys
import os
import json
from time import localtime, strftime, clock
import cPickle as pickle

# The CSS code generator matrix
G= np.matrix([[0,0,0,1,1,1,1], \
              [0,1,1,0,0,1,1], \
              [1,0,1,0,1,0,1]]).astype(np.int32)
error_keys= ['errX3', 'errX4', 'errZ3', 'errZ4']
syndrome_keys= ['synX12', 'synX34', 'synZ12', 'synZ34']

def syndrome(err):

    return np.dot(err, G.transpose()) % 2

def lookup_correction(syn):

    correction_index= np.dot(syn, [[4], [2], [1]]) - 1
    return y2indicator(correction_index, 7)

def lookup_correction_from_err(err):

    syn= syndrome(err)
    return lookup_correction(syn)

def find_logical_fault(err):

    syndrome= np.dot(G, err.transpose()) % 2
    correction_index= np.dot([[4, 2, 1]], syndrome.transpose()) - 1
    correction = y2indicator(correction_index, 7)
    coset= (err + correction) % 2
    logical_err= np.sum(coset) % 2
    return logical_err

def num_logical_fault(prediction, test_data):

    error_counter= 0.0
    for i in range(len(prediction[error_keys[0]])):
        for key in error_keys:
            if (find_logical_fault(\
                prediction[key][i] * np.ones(7) + test_data.rec[key][i] % 2)):
                error_counter+=1
                break
    return error_counter/len(prediction[error_keys[0]])

class Model:
    
    def __init__(self, data):
        raw_data, p, lu_avg, lu_std, data_size = get_data(datafolder + filename)
        total_size= np.shape(raw_data['synX12'])[0]
        test_size= int(test_fraction * total_size)
        train_data, test_data = io_data_factory(raw_data, test_size)
        self.total_size = total_size
        self.p = p
        self.lu_avg = lu_avg
        self.lu_std = lu_std
        self.data_size = data_size
        self.test_size = test_size
        self.train_data= train_data
        self.test_data = test_data
        self.train_size= total_size - test_size
        self.error_scale= 1.0*total_size/data_size

class Data:

    def __init__(self, data, train_mode=True):
        self.input= {}
        self.vir_output= {}
        self.vir_output_ind= {}

        synX= np.concatenate(\
            (data['synX12'], data['synX34']), axis= 1).reshape(-1, 2, 6)
        synZ= np.concatenate(\
            (data['synZ12'], data['synZ34']), axis= 1).reshape(-1, 2, 6)
        self.input['errX3']= synX
        self.input['errX4']= synX
        self.input['errZ3']= synZ
        self.input['errZ4']= synZ

        rep_X1= lookup_correction(data['synX12'][:,0:3])
        rep_X2= lookup_correction(data['synX12'][:,3:6])
        rep_Z1= lookup_correction(data['synZ12'][:,0:3])
        rep_Z2= lookup_correction(data['synZ12'][:,3:6])
        rep_X3= lookup_correction(data['synX34'][:,0:3])
        rep_X4= lookup_correction(data['synX34'][:,3:6])
        rep_Z3= lookup_correction(data['synZ34'][:,0:3])
        rep_Z4= lookup_correction(data['synZ34'][:,3:6])

        self.rec= {}
        self.rec['errX3']= (data['errX3'] + rep_X1 + \
            lookup_correction_from_err((rep_X1 + rep_X3) % 2)) % 2
        self.rec['errZ3']= (data['errZ3'] + rep_Z1 + rep_Z2 + \
            lookup_correction_from_err((rep_Z1 + rep_Z2 + rep_Z3) % 2)) % 2
        self.rec['errX4']= (data['errX4'] + rep_X1 + rep_X2 + \
            lookup_correction_from_err((rep_X1 + rep_X2 + rep_X4) % 2)) % 2    
        self.rec['errZ4']= (data['errZ4'] + rep_Z3 + \
            lookup_correction_from_err((rep_Z3 + rep_Z4) % 2)) % 2

        for key in error_keys:
            self.vir_output[key] = np.array(\
                np.sum((self.rec[key] +\
                lookup_correction_from_err(self.rec[key]))\
                 % 2, axis= 1) % 2).transpose()
            
        for key in error_keys:
            self.vir_output_ind[key]=\
            y2indicator(self.vir_output[key],2**1).astype(np.int8)
        
        if (train_mode):
            self.rec= None
        else:
            self.vir_output = None
            self.vir_output_ind= None
            
def io_data_factory(data, test_size):

    train_data_arg = {key:data[key][:-test_size,] for key in data.keys()}
    test_data_arg  = {key:data[key][-test_size:,] for key in data.keys()}
    train_data = Data(train_data_arg)
    test_data = Data(test_data_arg, False)
    return train_data, test_data

def get_data(filename):

    data= {}
    for key in syndrome_keys:
        data[key]= []
    for key in error_keys:
        data[key]= []
    with open(filename) as file:
        first_line = file.readline();
        p, lu_avg, lu_std, data_size = first_line.split(' ')
        p= float(p)
        lu_avg= float(lu_avg)
        lu_std= float(lu_std)
        data_size= int(data_size)
        for line in file.readlines():
            line_list= line.split(' ')
            data['synX12'].append([bit for bit in ''.join(line_list[0:2])])
            data['synX34'].append([bit for bit in ''.join(line_list[2:4])])
            data['synZ12'].append([bit for bit in ''.join(line_list[8:10])])
            data['synZ34'].append([bit for bit in ''.join(line_list[10:12])])
            data['errX3'].append([int(line_list[6],2)])
            data['errX4'].append([int(line_list[7],2)])
            data['errZ3'].append([int(line_list[14],2)])
            data['errZ4'].append([int(line_list[15],2)])
    for key in data.keys():
        data[key]= np.array(data[key]).astype(np.int8)
    return data, p, lu_avg, lu_std, data_size

def train(param, train_data, test_data, \
          num_classes, num_inputs, input_size, n_batches):

    prediction= {}
    verbose= param['usr']['verbose']
    batch_size= param['opt']['batch size']
    learning_rate= param['opt']['learning rate']
    num_iterations= param['opt']['iterations']
    momentum_val= param['opt']['momentum']
    decay_rate= param['opt']['decay']
    num_hidden= param['nn']['num hidden'] 
    W_std= param['nn']['W std'] 
    b_std= param['nn']['b std'] 

    # define all parts of the tf graph
    tf.reset_default_graph()
    x = {}
    y = {}
    lstm = {}
    lstmOut = {}
    W= {}
    b= {}
    logits= {}
    loss= {}
    predict= {}
    
    for key in error_keys:
        with tf.variable_scope(key):

            x[key] = tf.placeholder(tf.float32, [None, num_inputs, input_size])
            y[key] = tf.placeholder(tf.float32, [None, num_classes])
            lstm[key] = tf.contrib.rnn.LSTMCell(num_hidden)
            lstmOut[key], _ = tf.nn.dynamic_rnn(\
                lstm[key], x[key], dtype=tf.float32)
            W[key]= tf.Variable(\
                tf.random_normal([num_hidden,num_classes], stddev=W_std))
            b[key]= tf.Variable(tf.random_normal([num_classes], stddev=b_std))
            logits[key]= tf.matmul(lstmOut[key][:,-1,:], W[key]) + b[key]
            loss[key]= tf.nn.softmax_cross_entropy_with_logits(\
                logits=logits[key], labels=y[key])
            predict[key]= tf.argmax(logits[key], 1)
    
    cost= tf.reduce_sum(sum(loss[key] for key in error_keys))
    train = tf.train.RMSPropOptimizer(\
        learning_rate, decay=decay_rate, momentum=momentum_val).minimize(cost)
    init = tf.global_variables_initializer()
    costs= []

    with tf.Session() as session:
        if (verbose): print('session begins '),
        session.run(init)

        for i in range(num_iterations):
            if (verbose): print('.'),

            for j in range(n_batches):
                beg= j * batch_size
                end= j * batch_size + batch_size
                feed_dict={}
                for key in error_keys:
                    feed_dict[x[key]]= train_data.input[key][beg:end,]
                    feed_dict[y[key]]= train_data.vir_output_ind[key][beg:end,]
                session.run(train, feed_dict)

            if (verbose>1):
                feed_dict={}
                for key in error_keys:
                    feed_dict[x[key]]= test_data.input[key]
                    feed_dict[y[key]]= test_data.vir_output_ind[key]
                test_cost = session.run(cost, feed_dict)
                costs.append(test_cost)
        
        for key in error_keys:
            prediction[key] = session.run(predict[key], \
                feed_dict= {x[key]: test_data.input[key]})
        if (verbose): print(' session ends.')

    if (verbose>1):
        plt.plot(costs)
        plt.show()

    return num_logical_fault(prediction, test_data)

### Run an entire benchmark

param= {}
param['nn']= {}
param['opt']= {}
param['data']= {}
param['usr']= {}
param['nn']['num hidden']= 100
param['nn']['W std']= 10.0**(-1.02861985)
param['nn']['b std']= 0.0
param['opt']['batch size']= 1000
param['opt']['learning rate']= 10.0**(-3.84178815)
param['opt']['iterations']= 10
param['opt']['momentum']= 0.99
param['opt']['decay']= 0.98
param['data']['test fraction']= 0.1
param['usr']['verbose']= True
 
verbose= param['usr']['verbose']
test_fraction= param['data']['test fraction']
output= []
num_classes= 2
num_inputs= 2
input_size= 6

datafolder= '../Data/CNOTLabPickle/e-04/'
file_list= os.listdir(datafolder)

for filename in file_list:

#     with open(datafolder+ filename + '.dat', "wb") as output_file:
#         print("Reading data from " + filename)
#         model= Model(datafolder+ filename + '.dat')
#         pickle.dump(model, output_file)
#         continue

    with open(datafolder + filename, 'rb') as input_file:
        m = pickle.load(input_file)
    
    batch_size= param['opt']['batch size']
    n_batches = m.train_size // batch_size

    avg= train(param, m.train_data, m.test_data, \
        num_classes, num_inputs, input_size, n_batches)

    run_log= {}
    run_log['data']= {}
    run_log['opt']= {}
    run_log['res']= {}
    run_log['data']['path']= filename
    run_log['data']['fault scale']= m.error_scale
    run_log['data']['total data size']= m.total_size
    run_log['data']['test set size']= m.test_size
    run_log['opt']['batch size']= batch_size
    run_log['opt']['number of batches']= n_batches
    run_log['res']['p']= m.p
    run_log['res']['lu avg']= m.lu_avg
    run_log['res']['lu std']= m.lu_std
    run_log['res']['nn avg'] = m.error_scale * avg
    run_log['res']['nn std'] = 0
    output.append(run_log)

outfilename = strftime("%Y-%m-%d-%H-%M-%S", localtime())
f = open('../Reports/CNOTLab/' + outfilename + '.json', 'w')
f.write(json.dumps(output, indent=2))
f.close()
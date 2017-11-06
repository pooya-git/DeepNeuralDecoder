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

# The CSS code generator matrix
G= np.matrix([[0,0,0,1,1,1,1], \
              [0,1,1,0,0,1,1], \
              [1,0,1,0,1,0,1]]).astype(np.int32)
error_keys= ['errX3', 'errX4', 'errZ3', 'errZ4']
syndrome_keys= ['synX12', 'synX34', 'synZ12', 'synZ34']

class Data:

    def __init__(self, data):
        self.input= {}
        self.output= {}
        self.output_ind= {}
        self.input['errX3']= np.concatenate( \
            (data['synX12'], data['synX34']), axis= 1).reshape(-1, 2, 6)
        self.input['errX4']= np.concatenate( \
            (data['synX12'], data['synX34']), axis= 1).reshape(-1, 2, 6)
        self.input['errZ3']= np.concatenate( \
            (data['synZ12'], data['synZ34']), axis= 1).reshape(-1, 2, 6)
        self.input['errZ4']= np.concatenate( \
            (data['synZ12'], data['synZ34']), axis= 1).reshape(-1, 2, 6)
        for key in error_keys:
            self.output[key]= data[key]
        for key in error_keys:
            self.output_ind[key]=\
            y2indicator(data[key],2**7).astype(np.float32)

def io_data_factory(data, test_size):

    train_data_arg = {key:data[key][:-test_size,] for key in data.keys()}
    test_data_arg  = {key:data[key][-test_size:,] for key in data.keys()}
    train_data = Data(train_data_arg)
    test_data = Data(test_data_arg)
    return train_data, test_data

def find_logical_fault(recovery, err):

    p_binary= '{0:07b}'.format(recovery)
    t_binary= '{0:07b}'.format(int(err))
    err_list= [int(a!=b) for a, b in zip(p_binary, t_binary)]
    err= np.matrix(err_list).astype(np.int32)
    syndrome= np.dot(G, err.transpose()) % 2
    correction_index= np.dot([[4, 2, 1]], syndrome) - 1
    correction = y2indicator(correction_index, 7)
    coset= (err + correction) % 2
    logical_err= np.sum(coset) % 2
    return logical_err

def num_logical_fault(prediction, test):

    error_counter= 0.0
    for i in range(len(prediction[error_keys[0]])):
        for key in error_keys:
            if (find_logical_fault(prediction[key][i], test.output[key][i])):
                error_counter+=1
                break
    return error_counter/len(prediction[error_keys[0]])

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
        data[key]= np.array(data[key]).astype(np.float32)
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
                    feed_dict[y[key]]= train_data.output_ind[key][beg:end,]
                session.run(train, feed_dict)

        for key in error_keys:
            prediction[key] = session.run(predict[key], \
                feed_dict= {x[key]: test_data.input[key]})
        if (verbose): print(' session ends.')

    return num_logical_fault(prediction, test_data)

### Run an entire benchmark

param= {}
param['nn']= {}
param['opt']= {}
param['data']= {}
param['usr']= {}
param['nn']['num hidden']= 500
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
output= []
num_classes= 2**7
num_inputs= 2
input_size= 6

datafolder= '../e-04/'
file_list= os.listdir(datafolder)

for filename in file_list:
    # Read data and find how much null syndromes to assume for error_scale
    print("Reading data from " + filename)
    raw_data, p, lu_avg, lu_std, data_size = get_data(datafolder + filename)

    test_fraction= param['data']['test fraction']
    total_size= np.shape(raw_data['synX12'])[0]
    test_size= int(test_fraction * total_size)
    train_data, test_data = io_data_factory(raw_data, test_size)

    batch_size= param['opt']['batch size']
    train_size= total_size - test_size
    n_batches = train_size // batch_size
    error_scale= 1.0*total_size/data_size

    avg= train(param, train_data, test_data, \
        num_classes, num_inputs, input_size, n_batches)

    run_log= {}
    run_log['data']= {}
    run_log['opt']= {}
    run_log['res']= {}
    run_log['data']['path']= filename
    run_log['data']['fault scale']= error_scale
    run_log['data']['total data size']= total_size
    run_log['data']['test set size']= test_size
    run_log['opt']['batch size']= batch_size
    run_log['opt']['number of batches']= n_batches
    run_log['res']['p']= p
    run_log['res']['lu avg']= lu_avg
    run_log['res']['lu std']= lu_std
    run_log['res']['nn avg'] = error_scale * avg
    run_log['res']['nn std'] = 0
    output.append(run_log)

outfilename = strftime("%Y-%m-%d-%H-%M-%S", localtime())
f = open('Reports/' + outfilename + '.json', 'w')
f.write(json.dumps(output, indent=2))
f.close()

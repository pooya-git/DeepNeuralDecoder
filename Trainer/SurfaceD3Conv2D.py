# ------------------------------------------------------------------------------
# 
#    SurfaceD3 trainer. Uses a 2d CNN.
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
import matplotlib
import matplotlib.pyplot as plt

# The Surface code generator matrices and look up table
gZ = np.matrix([[1, 0, 0, 1, 0, 0, 0, 0, 0], \
                [0, 1, 1, 0, 1, 1, 0, 0, 0], \
                [0, 0, 0, 1, 1, 0, 1, 1, 0], \
                [0, 0, 0, 0, 0, 1, 0, 0, 1]]).astype(np.int32);
gX = np.matrix([[1, 1, 0, 1, 1, 0, 0, 0, 0], \
                [0, 0, 0, 0, 0, 0, 1, 1, 0], \
                [0, 1, 1, 0, 0, 0, 0, 0, 0], \
                [0, 0, 0, 0, 1, 1, 0, 1, 1]]).astype(np.int32);
XcorrectionMat = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0], \
                            [0, 0, 0, 0, 0, 0, 0, 0, 1], \
                            [0, 0, 0, 0, 0, 0, 1, 0, 0], \
                            [0, 0, 0, 0, 1, 1, 0, 0, 0], \
                            [0, 1, 0, 0, 0, 0, 0, 0, 0], \
                            [0, 0, 0, 0, 0, 1, 0, 0, 0], \
                            [0, 0, 0, 0, 1, 0, 0, 0, 0], \
                            [0, 0, 0, 0, 1, 0, 0, 0, 1], \
                            [1, 0, 0, 0, 0, 0, 0, 0, 0], \
                            [1, 0, 0, 0, 0, 0, 0, 0, 1], \
                            [0, 0, 0, 1, 0, 0, 0, 0, 0], \
                            [0, 0, 0, 1, 0, 0, 0, 0, 1], \
                            [1, 1, 0, 0, 0, 0, 0, 0, 0], \
                            [1, 0, 0, 0, 0, 1, 0, 0, 0], \
                            [1, 0, 0, 0, 1, 0, 0, 0, 0], \
                            [0, 0, 0, 1, 0, 1, 0, 0, 0]]).astype(np.int32);
ZcorrectionMat = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0], \
                            [0, 0, 0, 0, 0, 1, 0, 0, 0], \
                            [0, 0, 1, 0, 0, 0, 0, 0, 0], \
                            [0, 1, 0, 0, 1, 0, 0, 0, 0], \
                            [0, 0, 0, 0, 0, 0, 1, 0, 0], \
                            [0, 0, 0, 0, 0, 0, 0, 1, 0], \
                            [0, 0, 1, 0, 0, 0, 1, 0, 0], \
                            [0, 0, 1, 0, 0, 0, 0, 1, 0], \
                            [1, 0, 0, 0, 0, 0, 0, 0, 0], \
                            [0, 0, 0, 0, 1, 0, 0, 0, 0], \
                            [0, 1, 0, 0, 0, 0, 0, 0, 0], \
                            [0, 1, 0, 0, 0, 1, 0, 0, 0], \
                            [1, 0, 0, 0, 0, 0, 1, 0, 0], \
                            [1, 0, 0, 0, 0, 0, 0, 1, 0], \
                            [0, 1, 0, 0, 0, 0, 1, 0, 0], \
                            [0, 1, 0, 0, 0, 0, 0, 1, 0]]).astype(np.int32);
ZL = np.matrix([[1,0,0,0,1,0,0,0,1]]).astype(np.int32);
XL = np.matrix([[0,0,1,0,1,0,1,0,0]]).astype(np.int32);

err_keys= ['errX3', 'errZ3']
syn_keys= ['synX', 'synZ']

class Data:

    def __init__(self, data):
        self.input= {}
        self.output= {}
        self.output_ind= {}
        self.input['errX3']= data['synX'].reshape(-1, 3, 2, 2)
        self.input['errX3']= np.lib.pad(self.input['errX3'], \
            ((0,0), (0,0), (1,1), (1,1)), 'constant')
        self.input['errX3']= self.input['errX3'].transpose(0, 2, 3, 1)

        self.input['errZ3']= data['synZ'].reshape(-1, 3, 2, 2)
        self.input['errZ3']= np.lib.pad(self.input['errZ3'], \
            ((0,0), (0,0), (1,1), (1,1)), 'constant')
        self.input['errZ3']= self.input['errZ3'].transpose(0, 2, 3, 1)

        for key in err_keys:
            self.output[key]= data[key]
        for key in err_keys:
            self.output_ind[key]=y2indicator(data[key],2**9).astype(np.float32)

def io_data_factory(data, test_size):

    train_data_arg = {key:data[key][:-test_size,] for key in data.keys()}
    test_data_arg  = {key:data[key][-test_size:,] for key in data.keys()}
    train_data = Data(train_data_arg)
    test_data = Data(test_data_arg)
    return train_data, test_data

def find_logical_fault(recovery, err, key):

    p_binary= '{0:09b}'.format(recovery)
    t_binary= '{0:09b}'.format(int(err))
    err_list= [int(a!=b) for a, b in zip(p_binary, t_binary)]
    err= np.matrix(err_list).astype(np.int32)
    if (key=='errX3'):
        syndrome= np.dot(gZ, err.transpose()) % 2
        correction_index= np.asscalar(np.dot([[8, 4, 2, 1]], syndrome))
        correction = XcorrectionMat[correction_index,:]
        errFinal = (correction + err) % 2
        logical_err = np.dot(ZL, errFinal.transpose()) % 2
        return logical_err

    elif (key=='errZ3'):
        syndrome= np.dot(gX, err.transpose()) % 2
        correction_index= np.asscalar(np.dot([[8, 4, 2, 1]], syndrome))
        correction = ZcorrectionMat[correction_index,:]
        errFinal = (correction + err) % 2
        logical_err = np.dot(XL, errFinal.transpose()) % 2
        return logical_err

    else: print('Key not recognized.')

def num_logical_fault(prediction, test):

    error_counter= 0.0
    for i in range(len(prediction[err_keys[0]])):
        for key in err_keys:
            if (find_logical_fault(prediction[key][i],test.output[key][i],key)):
                error_counter+=1
                break
    return error_counter/len(prediction[err_keys[0]])

def get_data(filename):

    data= {}
    for key in syn_keys:
        data[key]= []
    for key in err_keys:
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
            data['synX'].append([bit for bit in ''.join(line_list[0:3])])
            data['synZ'].append([bit for bit in ''.join(line_list[6:9])])
            data['errX3'].append([int(line_list[5],2)])
            data['errZ3'].append([int(line_list[11],2)])
    for key in data.keys():
        data[key]= np.array(data[key]).astype(np.float32)
    return data, p, lu_avg, lu_std, data_size

def train(param, train_data, test_data, \
          num_classes, num_channels, kernel_size, num_filters, n_batches):

    prediction= {}
    verbose= param['usr']['verbose']
    batch_size= param['opt']['batch size']
    learning_rate= param['opt']['learning rate']
    num_iterations= param['opt']['iterations']
    momentum_val= param['opt']['momentum']
    decay_rate= param['opt']['decay']
    num_hiddens= param['nn']['num hidden'] 
    W_std= param['nn']['W std'] 
    b_std= param['nn']['b std'] 

    # define all parts of the tf graph
    tf.reset_default_graph()
    x = {}
    y = {}
    conv = {}
    pool = {}
    pool_flat = {}
    dense = {}
    logits= {}
    loss= {}
    predict= {}
    
    for key in err_keys:
        with tf.variable_scope(key):

            x[key] = tf.placeholder(tf.float32, [None, 4, 4, num_channels])
            y[key] = tf.placeholder(tf.float32, [None, num_classes])
            conv[key] = tf.layers.conv2d(\
                x[key], filters= num_filters, \
                kernel_size= kernel_size, \
                padding= 'same', activation=tf.nn.relu)
            pool[key] = tf.layers.max_pooling2d(conv[key], \
                pool_size=[2, 2], strides=1)
            pool_flat[key]= tf.reshape(pool[key], [-1, 3*3*num_filters])
            dense[key] = tf.layers.dense(inputs=pool_flat[key], \
                units=num_hiddens, activation=tf.nn.relu)
            logits[key] = tf.layers.dense(dense[key], units=num_classes)
            loss[key]= tf.nn.softmax_cross_entropy_with_logits(\
                logits=logits[key], labels=y[key])
            predict[key]= tf.argmax(logits[key], 1)
    
    cost= tf.reduce_sum(sum(loss[key] for key in err_keys))
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
                for key in err_keys:
                    feed_dict[x[key]]= train_data.input[key][beg:end,]
                    feed_dict[y[key]]= train_data.output_ind[key][beg:end,]
                session.run(train, feed_dict)
            
            if (verbose>1):
                feed_dict={}
                for key in err_keys:
                    feed_dict[x[key]]= test_data.input[key]
                    feed_dict[y[key]]= test_data.output_ind[key]
                test_cost = session.run(cost, feed_dict)
                costs.append(test_cost)

        for key in err_keys:
            prediction[key] = session.run(predict[key], \
                feed_dict= {x[key]: test_data.input[key]})
        if (verbose): print(' session ends.')

    if (verbose>1):
        plt.plot(costs)
        plt.show()
    return num_logical_fault(prediction, test_data)

param= {}
param['nn']= {}
param['opt']= {}
param['data']= {}
param['usr']= {}
param['nn']['num hidden']= 500
param['nn']['W std']= 10.0**(-3)
param['nn']['b std']= 0.0
param['opt']['batch size']= 1000
param['opt']['learning rate']= 10.0**(-5)
param['opt']['iterations']= 5
param['opt']['momentum']= 0.99
param['opt']['decay']= 0.99
param['data']['test fraction']= 0.1
param['usr']['verbose']= 2
 
verbose= param['usr']['verbose']
output= []
num_classes= 2**9
num_inputs= 3
kernel_size= 2
num_filters = 10
num_channels = 3

datafolder= '../SurfaceData/e-04/'
file_list= os.listdir(datafolder)

for filename in file_list:
    # Read data and find how much null syndromes to assume for error_scale
    print("Reading data from " + filename)
    raw_data, p, lu_avg, lu_std, data_size = get_data(datafolder + filename)

    test_fraction= param['data']['test fraction']
    total_size= np.shape(raw_data['synX'])[0]
    test_size= int(test_fraction * total_size)
    train_data, test_data = io_data_factory(raw_data, test_size)

    batch_size= param['opt']['batch size']
    train_size= total_size - test_size
    n_batches = train_size // batch_size
    error_scale= 1.0*total_size/data_size

    avg= train(param, train_data, test_data, \
        num_classes, num_inputs, kernel_size, num_filters, n_batches)

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
f = open('../Reports/SurfaceD3/' + outfilename + '.json', 'w')
f.write(json.dumps(output, indent=2))
f.close()
# ------------------------------------------------------------------------------
# 
#    SurfaceD3 trainer. Uses FF network to generate results of arXiv:1705.00857
#
#    Copyright (C) 2017 Pooya Ronagh
# 
# ------------------------------------------------------------------------------

from builtins import range
import numpy as np
import tensorflow as tf
import sys
import os
import json
from time import localtime, strftime, clock
import matplotlib
import matplotlib.pyplot as plt
import cPickle as pickle
from util import y2indicator
from SurfaceD3Data import *

def train(param, train_data, test_data, num_classes, n_batches):

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
    W1 = {}
    b1 = {}
    hidden= {}
    W2= {}
    b2= {}
    logits= {}
    loss= {}
    predict= {}
    
    for key in pauli_keys:
        with tf.variable_scope(key):

            x[key] = tf.placeholder(tf.float32, [None, 12])
            y[key] = tf.placeholder(tf.float32, [None, num_classes])
            W1[key]= tf.Variable(\
                tf.random_normal([12, num_hidden], stddev=W_std))
            b1[key]= tf.Variable(tf.random_normal([num_hidden], stddev=b_std))
            hidden[key]= tf.nn.sigmoid(tf.matmul(x[key], W1[key]) + b1[key])
            W2[key]= tf.Variable(\
                tf.random_normal([num_hidden, num_classes], stddev=W_std))
            b2[key]= tf.Variable(tf.random_normal([num_classes], stddev=b_std))
            logits[key]= tf.nn.sigmoid(tf.matmul(hidden[key], W2[key]) +b2[key])
            loss[key]= tf.nn.softmax_cross_entropy_with_logits(\
                logits=logits[key], labels=y[key])
            predict[key]= tf.argmax(logits[key], 1)
    
    cost= tf.reduce_sum(sum(loss[key] for key in pauli_keys))
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
                for key in pauli_keys:
                    feed_dict[x[key]]= train_data.syn[key][beg:end,]
                    feed_dict[y[key]]= train_data.log_1hot[key][beg:end,]
                session.run(train, feed_dict)
            
            if (verbose>1):
                feed_dict={}
                for key in pauli_keys:
                    feed_dict[x[key]]= test_data.syn[key]
                    feed_dict[y[key]]= test_data.log_1hot[key]
                test_cost = session.run(cost, feed_dict)
                costs.append(test_cost)

        for key in pauli_keys:
            prediction[key] = session.run(predict[key], \
                feed_dict= {x[key]: test_data.syn[key]})
        if (verbose): print(' session ends.')

    if (verbose>1):
        plt.plot(costs)
        plt.show()
    return num_logical_fault(prediction, test_data)

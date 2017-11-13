# ------------------------------------------------------------------------------
# 
#    SurfaceD3 trainer. Uses a 3d CNN.
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

def train(param, train_data, test_data, \
          num_classes, kernel_size, num_filters, n_batches):

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
    
    for key in pauli_keys:
        with tf.variable_scope(key):

            x[key] = tf.placeholder(tf.float32, [None, 3, 4, 4, 1])
            y[key] = tf.placeholder(tf.float32, [None, num_classes])
            conv[key] = tf.layers.conv3d(\
                x[key], filters= num_filters,\
                kernel_size= kernel_size,\
                padding= 'same', activation=tf.nn.relu)
            pool[key] = tf.layers.max_pooling3d(conv[key],\
                pool_size=[2, 2, 2], strides=1)
            pool_flat[key]= tf.reshape(pool[key], [-1, 2 * 3 * 3 * num_filters])
            dense[key] = tf.layers.dense(inputs=pool_flat[key],\
                units=num_hiddens, activation=tf.nn.relu)
            logits[key] = tf.layers.dense(dense[key], units=num_classes)
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
                    feed_dict[x[key]]= train_data.input[key][beg:end,]
                    feed_dict[y[key]]= train_data.log_1hot[key][beg:end,]
                session.run(train, feed_dict)
            
            if (verbose>1):
                feed_dict={}
                for key in pauli_keys:
                    feed_dict[x[key]]= test_data.input[key]
                    feed_dict[y[key]]= test_data.log_1hot[key]
                test_cost = session.run(cost, feed_dict)
                costs.append(test_cost)

        for key in pauli_keys:
            prediction[key] = session.run(predict[key], \
                feed_dict= {x[key]: test_data.input[key]})
        if (verbose): print(' session ends.')

    if (verbose>1):
        plt.plot(costs)
        plt.show()
    return num_logical_fault(prediction, test_data)
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
import sys, os, json
from time import localtime, strftime, clock
import cPickle as pickle
from ExRecCNOTData import *
from util import *

def train(m, num_classes, num_inputs, input_size, trial):

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
    pointer= m.test_size * trial
    test_beg= (m.train_size + pointer) % m.data_size
    test_end= (m.train_size + m.test_size + pointer) % m.data_size
    if not test_end: test_end = None

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
    
    for key in err_keys:
        with tf.variable_scope(key):

            x[key] = tf.placeholder(tf.float32, [None, num_inputs, input_size])
            y[key] = tf.placeholder(tf.float32, [None, num_classes])
            lstm[key] = tf.contrib.rnn.LSTMCell(num_hiddens)
            lstmOut[key], _ = tf.nn.dynamic_rnn(\
                lstm[key], x[key], dtype=tf.float32)
            W[key]= tf.Variable(\
                tf.random_normal([num_hiddens, num_classes], stddev=W_std))
            b[key]= tf.Variable(tf.random_normal([num_classes], stddev=b_std))
            logits[key]= tf.matmul(lstmOut[key][:,-1,:], W[key]) + b[key]
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

            for j in range(m.num_batches):
                beg= (j * batch_size + pointer) % m.data_size
                end= (j * batch_size + batch_size + pointer) % m.data_size
                if not end: end = None
                feed_dict={}
                for key in err_keys:
                    if (beg < end):
                        feed_dict[x[key]]= m.data.syn[key][beg:end]
                        feed_dict[y[key]]= m.data.err_1hot[key][beg:end]
                    else:
                        feed_dict[x[key]]= np.concatenate(\
                            (m.data.syn[key][beg:],\
                             m.data.syn[key][:end]), axis=0)
                        feed_dict[y[key]]= np.concatenate(\
                            (m.data.err_1hot[key][beg:],\
                             m.data.err_1hot[key][:end]), axis=0)
                session.run(train, feed_dict)

            if (verbose>1):
                feed_dict={}
                for key in err_keys:
                    feed_dict[x[key]]= m.data.syn[key][test_beg:test_end]
                    feed_dict[y[key]]= m.data.err_1hot[key][test_beg:test_end]
                test_cost = session.run(cost, feed_dict)
                costs.append(test_cost)
        
        for key in err_keys:
            prediction[key] = session.run(predict[key], \
                feed_dict= {x[key]: m.data.syn[key][test_beg:test_end]})
        if (verbose): print(' session ends.')

    if (verbose>1):
        plt.plot(costs)
        plt.show()

    return m.error_scale * num_logical_fault(\
        prediction,\
        {key: m.data.syn[key][test_beg:test_end] for key in m.data.syn.keys()},\
        {key: m.data.rec[key][test_beg:test_end] for key in m.data.rec.keys()})

if __name__== '__main__':

    param= {}
    param['nn']= {}
    param['opt']= {}
    param['data']= {}
    param['usr']= {}
    param['nn']['num hidden']= 500
    param['nn']['W std']= 10.0**(-0.95849325)
    param['nn']['b std']= 0.0
    param['opt']['batch size']= 1000
    param['opt']['learning rate']= 10.0**(-4.55693412)
    param['opt']['iterations']= 20
    param['opt']['momentum']= 0.99
    param['opt']['decay']= 0.98
    param['data']['test fraction']= 0.1
    param['data']['num trials']= 10
    param['usr']['verbose']= True
    param['nn']['type']= 'ExRecCNOTLabLSTM'

    verbose= param['usr']['verbose']
    test_fraction= param['data']['test fraction']
    batch_size= param['opt']['batch size']
    num_trials= param['data']['num trials']
    output= []
    num_classes= 2
    num_inputs= 2
    input_size= 6

    datafolder= '../Data/CNOTPkl/e-04/'
    file_list= os.listdir(datafolder)

    count= 0
    for filename in file_list:
        if count>5: break
        else: count+=1
        print datafolder

        with open(datafolder + filename, 'rb') as input_file:
            print("Pickling model from " + filename)
            m = pickle.load(input_file)

        m.test_size= int(test_fraction * m.data_size)
        m.train_size= m.data_size - m.test_size
        m.num_batches = m.train_size // batch_size
        m.error_scale= 1.0 * m.data_size / m.total_size
        m.param = param

        fault_rates= []
        for i in range(num_trials):
            fault_rates.append(train(m, num_classes, num_inputs, input_size, i))

        run_log= {}
        run_log['data']= {}
        run_log['opt']= {}
        run_log['res']= {}
        run_log['param']= param
        run_log['data']['path']= filename
        run_log['data']['fault scale']= m.error_scale
        run_log['data']['total size']= m.total_size
        run_log['data']['test size']= m.test_size
        run_log['data']['train size']= m.train_size
        run_log['opt']['batch size']= batch_size
        run_log['opt']['number of batches']= m.num_batches
        run_log['res']['p']= m.p
        run_log['res']['lu avg']= m.lu_avg
        run_log['res']['lu std']= m.lu_std
        run_log['res']['nn res'] = fault_rates
        run_log['res']['nn avg'] = np.mean(fault_rates)
        run_log['res']['nn std'] = np.std(fault_rates)
        output.append(run_log)

    outfilename = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    f = open('../Reports/CNOTLab/' + outfilename + '.json', 'w')
    f.write(json.dumps(output, indent=2))
    f.close()
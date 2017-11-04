# ------------------------------------------------------------------------------
# 
#    CNOTExRec trainer. Uses an RNN w/ 4 LSTM cells to train X & Z at same time.
#
#    Copyright (C) 2017 Pooya Ronagh
# 
# ------------------------------------------------------------------------------

from builtins import range
import numpy as np
import tensorflow as tf
import sys
from util import y2indicator

# The CSS code generator matrix
G= np.matrix([[0,0,0,1,1,1,1], \
              [0,1,1,0,0,1,1], \
              [1,0,1,0,1,0,1]]).astype(np.int32)

class Data:

    def __init__(self, syn12, syn34, errX3, errX4, errZ3, errZ4):
        self.input= np.concatenate((syn12, syn34), axis= 1).reshape(-1, 2, 12)
        # in_list= tf.transpose(self.in_arr, [1, 0, 2])
        # in_list= tf.reshape(in_list, [-1, 12])
        # in_list= tf.split(in_list, 2, 0)
        # self.input= self.in_list
        self.output= []
        self.output.append(errX3)
        self.output.append(errX4)
        self.output.append(errZ3)
        self.output.append(errZ4)
        self.output_ind= []
        self.output_ind.append(y2indicator(errX3, 2**7).astype(np.float32))
        self.output_ind.append(y2indicator(errX4, 2**7).astype(np.float32))
        self.output_ind.append(y2indicator(errZ3, 2**7).astype(np.float32))
        self.output_ind.append(y2indicator(errZ4, 2**7).astype(np.float32))

def io_data_factory(data, test_size):

    train_data = Data(data['syn12'][:-test_size,], data['syn34'][:-test_size,],\
        data['errX3'][:-test_size,], data['errX4'][:-test_size,], \
        data['errZ3'][:-test_size,], data['errZ4'][:-test_size,])
    test_data = Data(data['syn12'][-test_size:,], data['syn34'][-test_size:,], \
        data['errX3'][-test_size:,], data['errX4'][-test_size:,], \
        data['errZ3'][-test_size:,], data['errZ4'][-test_size:,])
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

def num_logical_fault(prediction, test_data):

    error_counter = 0.0
    for i in range(len(prediction[0])):
        for j in range(4):
            if (find_logical_fault(prediction[j][i], test_data.output[j][i])):
                error_counter+=1
                break
    return error_counter/len(prediction[0])
    
def get_data(filename):

    data= {}
    data['syn12']= []
    data['syn34']= []
    data['errX3']= []
    data['errX4']= []
    data['errZ3']= []
    data['errZ4']= []
    with open(filename) as file:
        first_line = file.readline();
        p, lu_avg, lu_std, data_size = first_line.split(' ')
        p= float(p)
        lu_avg= float(lu_avg)
        lu_std= float(lu_std)
        data_size= int(data_size)
        for line in file.readlines():
            line_list= line.split(' ')
            syn12_list= [line_list[i] for i in [0, 1, 8, 9]]
            syn34_list= [line_list[i] for i in [2, 3,10,11]]
            data['syn12'].append([bit for bit in ''.join(syn12_list)])
            data['syn34'].append([bit for bit in ''.join(syn34_list)])
            data['errX3'].append([int(line_list[6],2)])
            data['errX4'].append([int(line_list[7],2)])
            data['errZ3'].append([int(line_list[14],2)])
            data['errZ4'].append([int(line_list[15],2)])
    data['syn12']= np.array(data['syn12']).astype(np.float32)
    data['syn34']= np.array(data['syn34']).astype(np.float32)
    data['errX3']= np.array(data['errX3']).astype(np.float32)
    data['errX4']= np.array(data['errX4']).astype(np.float32)
    data['errZ3']= np.array(data['errZ3']).astype(np.float32)
    data['errZ4']= np.array(data['errZ4']).astype(np.float32)
    return data, p, lu_avg, lu_std, data_size

def train(filename, param):

    test_fraction= param['data']['test fraction']
    batch_size= param['data']['batch size']
    learning_rate= param['opt']['learning rate']
    num_iterations= param['opt']['iterations']
    momentum_val= param['opt']['momentum']
    decay_rate= param['opt']['decay']
    verbose= param['usr']['verbose']
    num_hidden= param['nn']['num hidden'] 

    output= {}
    output['data']= {}
    output['opt']= {}
    output['res']= {}

    # Read data and figure out how much null syndromes to assume for error_scale
    print("Reading data from " + filename)
    output['data']['path']= filename

    raw_data, p, lu_avg, lu_std, data_size = get_data(filename)
    output['res']['p']= p
    output['res']['lu avg']= lu_avg
    output['res']['lu std']= lu_std

    total_size= np.shape(raw_data['syn12'])[0]
    test_size= int(test_fraction * total_size)
    error_scale= 1.0*total_size/data_size
    output['data']['fault scale']= error_scale

    train_data, test_data = io_data_factory(raw_data, test_size)

    N, num_inputs, input_size= train_data.input.shape
    _, num_classes= train_data.output_ind[0].shape
    n_batches = N // batch_size

    output['data']['total data size']= total_size
    output['data']['test set size']= test_size
    output['opt']['batch size']= batch_size
    output['opt']['number of batches']= n_batches

    num_logical_errors= 4
    network= []
    tf.reset_default_graph()
    for i in range(num_logical_errors):
        network.append({})
        network[i]['x'] = tf.placeholder(tf.float32, [None, num_inputs, \
                                                             input_size])
        network[i]['y'] = tf.placeholder(tf.float32, [None, num_classes])
        network[i]['LSTM'] = tf.contrib.rnn.LSTMCell(num_hidden, reuse=(i>0))
        network[i]['LSTMOut'], _ = tf.nn.dynamic_rnn(network[i]['LSTM'], \
            network[i]['x'], dtype=tf.float32)
        network[i]['W']= tf.Variable(tf.random_normal([num_hidden,num_classes]))
        network[i]['b']= tf.Variable(tf.random_normal([num_classes]))
        network[i]['logits']= \
            tf.matmul(network[i]['LSTMOut'][:,-1,:], network[i]['W']) + \
            network[i]['b']
    
    loss= sum([tf.nn.softmax_cross_entropy_with_logits(\
        logits=network[i]['logits'], labels=network[i]['y']) \
        for i in range(num_logical_errors)])
    cost= tf.reduce_sum(loss)

    # Choose an optimizer
    train = tf.train.RMSPropOptimizer(learning_rate, \
        decay=decay_rate, momentum=momentum_val).minimize(cost)

    # This is the predict of the network in the active mode
    predict= []
    for i in range(num_logical_errors):
        predict.append(tf.argmax(network[i]['logits'], 1))

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        for i in range(num_iterations):

            # train all batches
            for j in range(n_batches):
                beg= j * batch_size
                end= j * batch_size + batch_size
                
                feed_dict={}
                for k in range(num_logical_errors):
                    feed_dict[network[k]['x']]= train_data.input[beg:end,]
                    feed_dict[network[k]['y']]= \
                                        train_data.output_ind[k][beg:end,]
                session.run(train, feed_dict)
        
        prediction= []
        for i in range(num_logical_errors):
            prediction.append(session.run(predict[i], \
                              feed_dict= {network[i]['x']: test_data.input}))

        avg= num_logical_fault(prediction, test_data)

        output['res']['nn avg'] = error_scale * avg
        output['res']['nn std'] = 0

    return output

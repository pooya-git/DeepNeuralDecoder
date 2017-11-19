# ------------------------------------------------------------------------------
# 
#    CNOTExRec trainer. Uses a feedforward network with 1 hidden layer.
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

class ioData:

    def __init__(self, syn, err):
        self.syn12 = syn[:,0:6]
        self.syn3 = syn[:,6:9]
        self.syn4 = syn[:,9:12]
        self.err3 = err[:,0]
        self.err4 = err[:,1]
        self.err3_ind = y2indicator(self.err3, 2**7)
        self.err4_ind = y2indicator(self.err4, 2**7)

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

def num_logical_fault(ErrX3_predict, ErrX4_predict, \
                      ErrZ3_predict, ErrZ4_predict,\
                      ErrX3_test, ErrX4_test, ErrZ3_test, ErrZ4_test):

    error_counter = 0
    for i in range(len(ErrX3_predict)):

        X3_fault= find_logical_fault(ErrX3_predict[i], ErrX3_test[i])
        X4_fault= find_logical_fault(ErrX4_predict[i], ErrX4_test[i])
        Z3_fault= find_logical_fault(ErrZ3_predict[i], ErrZ3_test[i])
        Z4_fault= find_logical_fault(ErrZ4_predict[i], ErrZ4_test[i])
        error_counter+= (X3_fault or X4_fault or Z3_fault or Z4_fault)

    return 1.0*error_counter/len(ErrX3_predict)

def get_data(filename):

    syn_X= []
    syn_Z= []
    err_X= []
    err_Z= []
    with open(filename) as file:
        first_line = file.readline();
        p, lu_avg, lu_std, data_size = first_line.split(' ')
        p= float(p)
        lu_avg= float(lu_avg)
        lu_std= float(lu_std)
        data_size= int(data_size) 
        for line in file.readlines()[1:]:
            line_list= line.split(' ')
            syn_X.append([bit for bit in line_list[0]])
            syn_Z.append([bit for bit in line_list[2]])
            err_X.append([int(line_list[1][0:7],2), int(line_list[1][7:14],2)])
            err_Z.append([int(line_list[3][0:7],2), int(line_list[3][7:14],2)])
    syn_X = np.matrix(syn_X).astype(np.int8)
    err_X = np.matrix(err_X).astype(np.float32)
    syn_Z = np.matrix(syn_Z).astype(np.int8)
    err_Z = np.matrix(err_Z).astype(np.float32)
    return syn_X, err_X, syn_Z, err_Z, p, lu_avg, lu_std, data_size

def train(filename, param, nn):

    test_fraction= param['data']['test fraction']
    batch_size= param['data']['batch size']
    learning_rate= param['opt']['learning rate']
    num_iterations= param['opt']['iterations']
    momentum_val= param['opt']['momentum']
    decay_rate= param['opt']['decay']
    verbose = param['usr']['verbose']

    output= {}
    output['data']= {}
    output['opt']= {}
    output['res']= {}

    # Read data and figure out how much null syndromes to assume for error_scale
    print("Reading data from " + filename)
    output['data']['path']= filename

    SynX, ErrX, SynZ, ErrZ, \
    p, lu_avg, lu_std, data_size = get_data(filename)
    output['res']['p']= p
    output['res']['lu avg']= lu_avg
    output['res']['lu std']= lu_std

    total_size= np.shape(SynX)[0]
    test_size= int(test_fraction * total_size)
    error_scale= 1.0*total_size/data_size
    output['data']['fault scale']= error_scale

    trainX = ioData(SynX[:-test_size,], ErrX[:-test_size,])
    testX = ioData(SynX[-test_size:,], ErrX[-test_size:,])
    testZ = ioData(SynZ[-test_size:,], ErrZ[-test_size:,])

    N, _ = trainX.syn12.shape
    n_batches = N // batch_size

    output['data']['total data size']= total_size
    output['data']['test set size']= test_size
    output['opt']['batch size']= batch_size
    output['opt']['number of batches']= n_batches

    # TF IO placehoders
    In= {}
    Out= {}
    for key in nn.keys():
        if nn[key]['type'] == 'input':
            In[key]= tf.placeholder(tf.float32, \
                shape= (None, nn[key]['dim']), name= key)
    for key in nn.keys():
        if nn[key]['type'] == 'output':
            Out[key]= tf.placeholder(tf.float32, \
                shape= (None, nn[key]['dim']), name= key)

    # TF weight and bias variables
    W= {}
    b= {}
    for beg in nn.keys():
        for end in nn[beg]['edges'].keys():
            init_biases= np.zeros(nn[end]['dim'])
            init_weights= np.random.randn(nn[beg]['dim'], nn[end]['dim']) * \
                nn[beg]['edges'][end]['w_std']
            b[(beg, end)]= tf.Variable(init_biases.astype(np.float32))
            W[(beg, end)]= tf.Variable(init_weights.astype(np.float32))

    # Feedforward rules
    H = tf.nn.relu(tf.matmul(In['Syn12'], W[('Syn12', 'H')]) +\
        b[('Syn12', 'H')])
    Err3 = tf.matmul(H, W[('H', 'Err3')]) + b[('H', 'Err3')] \
        + tf.matmul(In['Syn3'], W[('Syn3', 'Err3')]) + b[('Syn3', 'Err3')]
    Err4 = tf.matmul(H, W[('H', 'Err4')]) + b[('H', 'Err4')] \
        + tf.matmul(In['Syn4'], W[('Syn4', 'Err4')]) + b[('Syn4', 'Err4')]
    
    # softmax_cross_entropy_with_logits take in the "logits"
    # if you wanted to know the actual output of the neural net,
    # you could pass "Yish" into tf.nn.softmax(logits)
    cost = tf.reduce_sum(\
    tf.nn.softmax_cross_entropy_with_logits(logits=Err3, labels=Out['Err3'])\
    + tf.nn.softmax_cross_entropy_with_logits(logits=Err4, labels=Out['Err4']))

    # Choose an optimizer
    train_op = tf.train.RMSPropOptimizer(learning_rate, \
        decay=decay_rate, momentum=momentum_val).minimize(cost)

    # This is the predict of the network in the active mode
    predict_Err3 = tf.argmax(Err3, 1)
    predict_Err4 = tf.argmax(Err4, 1)

    costs = []
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        for i in range(num_iterations):

            # train all batches
            for j in range(n_batches):
                beg= j * batch_size
                end= j * batch_size + batch_size
                
                Syn12_batch = trainX.syn12[beg : end,]
                Syn3_batch = trainX.syn3[beg : end,]
                Syn4_batch = trainX.syn4[beg : end,]
                Err3_batch = trainX.err3_ind[beg : end,]
                Err4_batch = trainX.err4_ind[beg : end,]

                session.run(train_op, \
                    feed_dict={ In['Syn12']: Syn12_batch, \
                                In['Syn3']: Syn3_batch, In['Syn4']: Syn4_batch,\
                                Out['Err3']:Err3_batch, Out['Err4']:Err4_batch})

            # do a test in the active mode
            if (verbose):
                test_cost = session.run(cost, \
                    feed_dict={ In['Syn12']: Syn12_batch, \
                                In['Syn3']: Syn3_batch, In['Syn4']: Syn4_batch,\
                                Out['Err3']:Err3_batch, Out['Err4']:Err4_batch})
                costs.append(test_cost)
                print("Iteration= ", i, ", Cost= ", test_cost)
                sys.stdout.flush()

        ErrX3_predict = session.run(predict_Err3, \
            feed_dict= {In['Syn12']: testX.syn12, \
                        In['Syn3']: testX.syn3, \
                        In['Syn4']: testX.syn4})
        ErrX4_predict = session.run(predict_Err4, \
            feed_dict= {In['Syn12']: testX.syn12, \
                        In['Syn3']: testX.syn3, \
                        In['Syn4']: testX.syn4})
        ErrZ3_predict = session.run(predict_Err3, \
            feed_dict= {In['Syn12']: testZ.syn12, \
                        In['Syn3']: testZ.syn3, \
                        In['Syn4']: testZ.syn4})
        ErrZ4_predict = session.run(predict_Err4, \
            feed_dict= {In['Syn12']: testZ.syn12, \
                        In['Syn3']: testZ.syn3, \
                        In['Syn4']: testZ.syn4})

        fault = error_scale * num_logical_fault( \
            ErrX3_predict, ErrX4_predict, ErrZ3_predict, ErrZ4_predict,\
            testX.err3, testX.err4, testZ.err3, testZ.err4)

        output['res']['nn'] = fault

    return output

'''
__main__():
  Args: 
    json parameter file,
    json network connectivity graph,
    data folder.
'''

if __name__ == '__main__':

    import sys
    import os
    import json
    from time import localtime, strftime

    with open(sys.argv[1]) as paramfile:
        param = json.load(paramfile)
    with open(sys.argv[2]) as networkfile:
        network = json.load(networkfile)
    datafolder= sys.argv[3]

    output= []

    for filename in os.listdir(datafolder):
        output.append(train(datafolder + filename, param, network))

    outfilename = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    f = open('Reports/' + outfilename + '.json', 'w')
    f.write(json.dumps(output, indent=2))
    f.close()


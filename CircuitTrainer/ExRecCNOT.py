#
# Author: Pooya Ronagh (2017)
# All rights reserved.
#
# 2-hidden layer NN in TensorFlow for the ExRecCNOT
#

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

def train(filename, param):

    test_fraction= param['data']['test fraction']
    batch_size= param['data']['batch size']
    learning_rate= param['opt']['learning rate']
    num_iterations= param['opt']['iterations']
    momentum_val= param['opt']['momentum']
    decay_rate= param['opt']['decay']
    num_h1= param['nn']['h1 size']
    num_h2= param['nn']['h2 size']
    init_w_std= param['nn']['init w-std']
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
    _, K = trainX.err3_ind.shape
    n_batches = N // batch_size

    output['data']['total data size']= total_size
    output['data']['test set size']= test_size
    output['opt']['batch size']= batch_size
    output['opt']['number of batches']= n_batches

    # TF IO placehoders
    Syn12 = tf.placeholder(tf.float32, shape=(None, 6), name='Syn12')
    Syn3 = tf.placeholder(tf.float32, shape=(None, 3), name='Syn3')
    Syn4 = tf.placeholder(tf.float32, shape=(None, 3), name='Syn3')
    Err3 = tf.placeholder(tf.float32, shape=(None, K), name='Err3')
    Err4 = tf.placeholder(tf.float32, shape=(None, K), name='Err4')

    # TF weights initial values
    W12 = np.random.randn(6, num_h1) * init_w_std
    W1h = np.random.randn(num_h1, K) / np.sqrt(num_h1)
    W2h = np.random.randn(num_h1, K) / np.sqrt(num_h1)
    W3 = np.random.randn(3, K) * init_w_std
    W4 = np.random.randn(3, K) * init_w_std

    # TF biases initial values
    b12 = np.zeros(num_h1)    
    b1h = np.zeros(K)
    b2h = np.zeros(K)
    b3 = np.zeros(K)  
    b4 = np.zeros(K)  

    # TF weights and biases variables
    W12 = tf.Variable(W12.astype(np.float32))
    b12 = tf.Variable(b12.astype(np.float32))
    W1h = tf.Variable(W1h.astype(np.float32))
    b1h = tf.Variable(b1h.astype(np.float32))
    W2h = tf.Variable(W2h.astype(np.float32))
    b2h = tf.Variable(b2h.astype(np.float32))
    W3 = tf.Variable(W3.astype(np.float32))
    b3 = tf.Variable(b3.astype(np.float32))
    W4 = tf.Variable(W4.astype(np.float32))
    b4 = tf.Variable(b4.astype(np.float32))

    # Feedforward rules
    Z1 = tf.nn.relu(tf.matmul(Syn12, W12) + b12)
    Err3_ish = tf.matmul(Z1, W1h) + b1h + tf.matmul(Syn3, W3) + b3
    Err4_ish = tf.matmul(Z1, W2h) + b2h + tf.matmul(Syn4, W4) + b4
    
    # softmax_cross_entropy_with_logits take in the "logits"
    # if you wanted to know the actual output of the neural net,
    # you could pass "Yish" into tf.nn.softmax(logits)
    cost = tf.reduce_sum(\
        tf.nn.softmax_cross_entropy_with_logits(logits=Err3_ish, labels=Err3)\
        + tf.nn.softmax_cross_entropy_with_logits(logits=Err4_ish, labels=Err4))

    # Choose an optimizer
    train_op = tf.train.RMSPropOptimizer(learning_rate, \
        decay=decay_rate, momentum=momentum_val).minimize(cost)

    # This is the predict of the network in the active mode
    predict_Err3 = tf.argmax(Err3_ish, 1)
    predict_Err4 = tf.argmax(Err4_ish, 1)

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
                    feed_dict={ Syn12: Syn12_batch, \
                                Syn3: Syn3_batch, Syn4: Syn4_batch, \
                                Err3: Err3_batch, Err4: Err4_batch})

            # do a test in the active mode
            if (verbose):
                test_cost = session.run(cost, \
                    feed_dict={ Syn12: Syn12_batch, \
                                Syn3: Syn3_batch, Syn4: Syn4_batch, \
                                Err3: Err3_batch, Err4: Err4_batch})
                costs.append(test_cost)
                print("Iteration= ", i, ", Cost= ", test_cost)
                sys.stdout.flush()

        ErrX3_predict = session.run(predict_Err3, \
            feed_dict= {Syn12: testX.syn12, Syn3: testX.syn3, Syn4: testX.syn4})
        ErrX4_predict = session.run(predict_Err4, \
            feed_dict= {Syn12: testX.syn12, Syn3: testX.syn3, Syn4: testX.syn4})
        ErrZ3_predict = session.run(predict_Err3, \
            feed_dict= {Syn12: testZ.syn12, Syn3: testZ.syn3, Syn4: testZ.syn4})
        ErrZ4_predict = session.run(predict_Err4, \
            feed_dict= {Syn12: testZ.syn12, Syn3: testZ.syn3, Syn4: testZ.syn4})

        fault = error_scale * num_logical_fault( \
            ErrX3_predict, ErrX4_predict, ErrZ3_predict, ErrZ4_predict,\
            testX.err3, testX.err4, testZ.err3, testZ.err4)

        output['res']['nn'] = fault

    return output

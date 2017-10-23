# 2-hidden layer NN in TensorFlow
# Author: Pooya Ronagh
# All rights reserved.

from __future__ import print_function, division
from builtins import range
import numpy as np
import tensorflow as tf
import sys

import matplotlib.pyplot as plt

# The CSS code generator matrix
G= np.matrix([[0,0,0,1,1,1,1], \
              [0,1,1,0,0,1,1], \
              [1,0,1,0,1,0,1]]).astype(np.int32)

def error_rate(p, t):

    error_counter = 0
    for i in range(len(p)):
        p_binary= '{0:014b}'.format(p[i])
        t_binary= '{0:014b}'.format(int(t[i]))
        err_list= [int(a!=b) for a, b in zip(p_binary, t_binary)]
        
        err3= np.matrix(err_list[0:7]).astype(np.int32)
        syndrome3= np.dot(G, err3.transpose()) % 2
        correction_index3= np.dot([[4, 2, 1]], syndrome3) - 1
        correction3 = y2indicator(correction_index3, 7)
        coset3= (err3 + correction3) % 2
        logical_err3= np.sum(coset3) % 2

        err4= np.matrix(err_list[7:]).astype(np.int32)
        syndrome4= np.dot(G, err4.transpose()) % 2
        correction_index4= np.dot([[4, 2, 1]], syndrome4) - 1
        correction4 = y2indicator(correction_index4, 7)
        coset4= (err4 + correction4) % 2
        logical_err4= np.sum(coset4) % 2

        if (logical_err3==1 or logical_err4==1):
           error_counter+=1

    return 1.0*error_counter/len(p)

def y2indicator(y, width):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, width)).astype(np.int32)
    for i in range(N):
        if (y[i] >= 0):
            ind[i, y[i]] = 1
    return ind

def get_data(filename):

    X= []
    Y= []
    with open(filename) as file:
        for line in file.readlines():
            X.append([x for x in line[0:12]])
            Y.append([int(''.join(line[12:-1]), 2)])
    X = np.matrix(X).astype(np.float32)
    Y = np.matrix(Y).astype(np.float32)
    return X, Y

def main(filename, data_size, test_size, batch_size):

    # Read data and figure out how much null syndromes to assume for error_scale
    print("Reading data ...")
    X, Y = get_data(filename)
    print("Data contains X ", np.shape(X), "and Y ", np.shape(Y))
    error_scale= np.shape(X)[0]/int(data_size)
    print("Scaling of error rates is ", error_scale)

    max_iter = 50
    print_period = 500

    lr = 0.00004
    reg = 0.01

    test_size= int(test_size)
    Xtrain_bcnot = X[:-test_size,0:6]
    Xtrain_acnot = X[:-test_size,6:]
    Ytrain = Y[:-test_size]
    Xtest_bcnot  = X[-test_size:,0:6]
    Xtest_acnot  = X[-test_size:,6:]
    Ytest  = Y[-test_size:]
    Ytrain_ind = y2indicator(Ytrain, 2**14)
    Ytest_ind = y2indicator(Ytest, 2**14)

    N, D_bcnot = Xtrain_bcnot.shape
    _, D_acnot = Xtest_acnot.shape
    _, K = Ytrain_ind.shape
    batch_size = int(batch_size)
    n_batches = N // batch_size

    print("Training set size= ", np.shape(Ytrain)[0])
    print("Test set size= ", np.shape(Ytest)[0])
    print("Batch size = ", batch_size)
    print("Number of batches = ", n_batches)

    # Number of nodes in each hidden layer
    M1 = 50
    M2 = 50
    
    # The input layer, joins the hidden layers in two stages:
    # Stage 1: EC1 and EC2 blocks feed to the first hidden layer
    # Stage 2: EC3 and EC4 blocks feed to the second hidden layer
    W1_init_bcnot = np.random.randn(D_bcnot, M1) / 100
    b1_init_bcnot = np.zeros(M1)
    W1_init_acnot = np.random.randn(D_acnot, M1) / 100
    b1_init_acnot = np.zeros(M1)
    
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2)
    b3_init = np.zeros(K)

    # TF placehoders and variables
    X_bcnot = tf.placeholder(tf.float32, shape=(None, D_bcnot), name='X_bcnot')
    X_acnot = tf.placeholder(tf.float32, shape=(None, D_acnot), name='X_acnot')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    W1_bcnot = tf.Variable(W1_init_bcnot.astype(np.float32))
    b1_bcnot = tf.Variable(b1_init_bcnot.astype(np.float32))
    W1_acnot = tf.Variable(W1_init_acnot.astype(np.float32))
    b1_acnot = tf.Variable(b1_init_acnot.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    # Feedforward rules
    Z1 = tf.nn.relu(tf.matmul(X_bcnot, W1_bcnot) + b1_bcnot)
    Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2 + \
                    tf.matmul(X_acnot, W1_acnot) + b1_acnot)
    Yish = tf.matmul(Z2, W3) + b3

    # softmax_cross_entropy_with_logits take in the "logits"
    # if you wanted to know the actual output of the neural net,
    # you could pass "Yish" into tf.nn.softmax(logits)
    cost = tf.reduce_sum(\
           tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=T))

    # Choose an optimizer
    train_op = tf.train.RMSPropOptimizer(\
               lr, decay=0.99, momentum=0.9).minimize(cost)

    # This is the predict of the network in the active mode
    predict_op = tf.argmax(Yish, 1)

    costs = []
    init = tf.global_variables_initializer()

    print('Training ...')
    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):

            # train all batches
            for j in range(n_batches):
                Xbatch_bcnot = Xtrain_bcnot[j * batch_size :\
                                           (j * batch_size + batch_size),]
                Xbatch_acnot = Xtrain_acnot[j * batch_size :\
                                           (j * batch_size + batch_size),]
                Ybatch = Ytrain_ind[j*batch_size:(j * batch_size + batch_size),]
                session.run(train_op, \
                    feed_dict={ X_bcnot: Xbatch_bcnot, \
                                X_acnot: Xbatch_acnot, T: Ybatch})

            # do a test in the active mode
            test_cost = session.run(cost, feed_dict={X_bcnot: Xtest_bcnot, \
                                                     X_acnot: Xtest_acnot, \
                                                     T: Ytest_ind})
            prediction = session.run(predict_op, \
                                          feed_dict={X_bcnot: Xtest_bcnot, \
                                                     X_acnot: Xtest_acnot})
            err = error_scale * error_rate(prediction, Ytest)
            print("Iteration = ", i, ", Cost = ", test_cost, ", Error = ", err)
            costs.append(test_cost)

    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

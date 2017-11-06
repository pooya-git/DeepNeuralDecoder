# ------------------------------------------------------------------------------
# 
#    CNOTExRec trainer. Uses an RNN with two cells to train on X data.
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
        self.syn = np.array([syn[:,0:12]]).reshape(-1, 2, 6)
        np.set_printoptions(threshold='nan')
        self.err3 = err[:,2]
        self.err4 = err[:,3]
        self.err3_ind = y2indicator(self.err3, 2**7).reshape(-1, 128)
        self.err4_ind = y2indicator(self.err4, 2**7).reshape(-1, 128)

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

    error_counter = []
    for i in range(len(ErrX3_predict)):

        X3_fault= find_logical_fault(ErrX3_predict[i], ErrX3_test[i])
        X4_fault= find_logical_fault(ErrX4_predict[i], ErrX4_test[i])
        Z3_fault= find_logical_fault(ErrZ3_predict[i], ErrZ3_test[i])
        Z4_fault= find_logical_fault(ErrZ4_predict[i], ErrZ4_test[i])
        error_counter.append(X3_fault or X4_fault or Z3_fault or Z4_fault)
    
    return np.mean(error_counter)
    
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
        for line in file.readlines():
            line_list= line.split(' ')
            syn_X.append([bit for bit in ''.join(line_list[0:4])])
            err_X.append([int(err,2) for err in line_list[4:8]])
            syn_Z.append([bit for bit in ''.join(line_list[8:12])])
            err_Z.append([int(err,2) for err in line_list[12:16]])
    syn_X = np.array(syn_X).astype(np.int8)
    err_X = np.array(err_X).astype(np.float32)
    syn_Z = np.array(syn_Z).astype(np.int8)
    err_Z = np.array(err_Z).astype(np.float32)
    return syn_X, err_X, syn_Z, err_Z, p, lu_avg, lu_std, data_size

def train(filename, param):

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

    N, _, num_inputs = trainX.syn.shape
    n_batches = N // batch_size

    output['data']['total data size']= total_size
    output['data']['test set size']= test_size
    output['opt']['batch size']= batch_size
    output['opt']['number of batches']= n_batches

    # TF IO placehoders
    num_inputs = 2
    num_hidden = 30
    num_outputs = 1
    num_classes= 128

    tf.reset_default_graph()

    In= tf.placeholder(tf.float32, [None, 2, 6])
    Err3= tf.placeholder(tf.float32, [None, num_classes])
    Err4= tf.placeholder(tf.float32, [None, num_classes])

    LSTM3 = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    Out3, _ = tf.nn.dynamic_rnn(LSTM3, In, dtype=tf.float32)
    init_biases= np.zeros(num_classes)
    init_weights= np.random.randn(num_hidden, num_classes)
    biases= tf.Variable(init_biases.astype(np.float32))
    weights= tf.Variable(init_weights.astype(np.float32))
    Err3ish= tf.matmul(Out3[:,-1], weights) + biases

    LSTM4 = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, reuse=True)
    Out4, _ = tf.nn.dynamic_rnn(LSTM4, In, dtype=tf.float32)
    init_biases= np.zeros(num_classes)
    init_weights= np.random.randn(num_hidden, num_classes)
    biases= tf.Variable(init_biases.astype(np.float32))
    weights= tf.Variable(init_weights.astype(np.float32))
    Err4ish= tf.matmul(Out4[:,-1], weights) + biases

    cost= tf.reduce_sum(\
          tf.nn.softmax_cross_entropy_with_logits(logits=Err3ish, labels=Err3)\
        + tf.nn.softmax_cross_entropy_with_logits(logits=Err4ish, labels=Err4))

    # Choose an optimizer
    train = tf.train.RMSPropOptimizer(learning_rate, \
        decay=decay_rate, momentum=momentum_val).minimize(cost)

    # This is the predict of the network in the active mode
    predict_Err3 = tf.argmax(Err3ish, 1)
    predict_Err4 = tf.argmax(Err4ish, 1)

    costs = []
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        for i in range(num_iterations):

            # train all batches
            for j in range(n_batches):
                beg= j * batch_size
                end= j * batch_size + batch_size
                
                Syn_batch = trainX.syn[beg : end,]
                Err3_batch = trainX.err3_ind[beg : end,]
                Err4_batch = trainX.err4_ind[beg : end,]

                session.run(train, \
                feed_dict={ In: Syn_batch, Err3: Err3_batch, Err4: Err4_batch})

        ErrX3_predict = session.run(predict_Err3, feed_dict= {In: testX.syn})
        ErrX4_predict = session.run(predict_Err4, feed_dict= {In: testX.syn})
        ErrZ3_predict = session.run(predict_Err3, feed_dict= {In: testZ.syn})
        ErrZ4_predict = session.run(predict_Err4, feed_dict= {In: testZ.syn})

        avg= num_logical_fault( \
            ErrX3_predict, ErrX4_predict, ErrZ3_predict, ErrZ4_predict,\
            testX.err3, testX.err4, testZ.err3, testZ.err4)

        output['res']['nn avg'] = error_scale * avg
        output['res']['nn std'] = 0

    return output

'''
__main__():
  Args: 
    json parameter file,
    data folder.
'''

if __name__ == '__main__':

    import sys
    import os
    import json
    from time import localtime, strftime

    with open(sys.argv[1]) as paramfile:
        param = json.load(paramfile)
    datafolder= sys.argv[2]

    output= []

    for filename in os.listdir(datafolder):
        output.append(train(datafolder + filename, param))

    outfilename = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    f = open('Reports/' + outfilename + '.json', 'w')
    f.write(json.dumps(output, indent=2))
    f.close()


# ------------------------------------------------------------------------------
# 
#    CNOTExRec trainer. Uses a feedforward network with latent input layers.
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

def syndrome(err):

    return np.dot(err, G.transpose()) % 2

def lookup_correction(syn):

    correction_index= np.dot(syn, [[4], [2], [1]]) - 1
    return y2indicator(correction_index, 7)

def lookup_correction_from_err(err):

    syn= syndrome(err)
    return lookup_correction(syn)

class dataConvertor:

    def __init__(self, syn_X, err_X, syn_Z, err_Z, \
                 save_flag= False, output_file= None):
        self.syn_12= np.concatenate((syn_X[:,0:6], syn_Z[:,0:6]), axis= 1)
        self.syn_34= np.concatenate((syn_X[:,6:12], syn_Z[:,6:12]), axis= 1)
        self.err_X3 = err_X[:,14:21]
        self.err_X4 = err_X[:,21:28]
        self.err_Z3 = err_Z[:,14:21]
        self.err_Z4 = err_Z[:,21:28]
        self.find_recovery()
        if (save_flag): self.save_data(output_file)

    def save_data(self, output_file):
        instream= open(output_file+'.txt', 'w+')
        for i in range(self.syn_12.shape[0]):
            instream.write(str(''.join([str(self.syn_12[i, j]) for j in\
                range(self.syn_12.shape[1])]))+ ' ') 
            instream.write(str(''.join([str(self.syn_34[i, j]) for j in\
                range(self.syn_34.shape[1])]))+ ' ') 
            instream.write(str(''.join([str(self.rec_bin[i, j]) for j in\
                range(self.rec_bin.shape[1])]))+ '\n')
        instream.close()

    def find_recovery(self):
        rep_X1= lookup_correction(self.syn_12[:,0:3])
        rep_X2= lookup_correction(self.syn_12[:,3:6])
        rep_Z1= lookup_correction(self.syn_12[:,6:9])
        rep_Z2= lookup_correction(self.syn_12[:,9:12])
        rep_X3= lookup_correction(self.syn_34[:,0:3])
        rep_X4= lookup_correction(self.syn_34[:,3:6])
        rep_Z3= lookup_correction(self.syn_34[:,6:9])
        rep_Z4= lookup_correction(self.syn_34[:,9:12])
        rec_X3= (self.err_X3 + rep_X1 + \
            lookup_correction_from_err((rep_X1 + rep_X3) % 2)) % 2
        rec_Z3= (self.err_Z3 + rep_Z1 + rep_Z2 + \
            lookup_correction_from_err((rep_Z1 + rep_Z2 + rep_Z3) % 2)) % 2
        rec_X4= (self.err_X4 + rep_X1 + rep_X2 + \
            lookup_correction_from_err((rep_X1 + rep_X2 + rep_X4) % 2)) % 2    
        rec_Z4= (self.err_Z4 + rep_Z3 + \
            lookup_correction_from_err((rep_Z3 + rep_Z4) % 2)) % 2
        rec = [rec_X3, rec_Z3, rec_X4, rec_Z4]
        self.rec_bin = np.array(\
            [np.sum((elt + lookup_correction_from_err(elt)) % 2, axis= 1) % 2\
            for elt in rec]).transpose()

def error_rate(prediction, recovery):
    return np.mean(prediction != recovery)

def get_data_from_full_format(filename):

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
            err_X.append([bit for bit in ''.join(line_list[4:8])])
            syn_Z.append([bit for bit in ''.join(line_list[8:12])])
            err_Z.append([bit for bit in ''.join(line_list[12:16]).strip('\n')])
    syn_X = np.array(syn_X).astype(np.int8)
    err_X = np.array(err_X).astype(np.int8)
    syn_Z = np.array(syn_Z).astype(np.int8)
    err_Z = np.array(err_Z).astype(np.int8)
    return syn_X, err_X, syn_Z, err_Z, p, lu_avg, lu_std, data_size

def convert_data(filename):

    print("Reading data from " + filename)
    SynX, ErrX, SynZ, ErrZ, \
    p, lu_avg, lu_std, data_size = get_data_from_full_format(filename)
    ioData(SynX, ErrX, SynZ, ErrZ, True, str(p))
    return output


def get_data(filename):

    syn12= []
    syn34= []
    fault= []
    with open(filename) as file:
        first_line = file.readline();
        p, lu_avg, lu_std, data_size = first_line.split(' ')
        p= float(p)
        lu_avg= float(lu_avg)
        lu_std= float(lu_std)
        data_size= int(data_size) 
        for line in file.readlines():
            line_list= line.split(' ')
            syn12.append([bit for bit in ''.join(line_list[0])])
            syn34.append([bit for bit in ''.join(line_list[1])])
            fault.append([bit for bit in ''.join(line_list[2]).strip('\n')])
    syn12 = np.array(syn12).astype(np.int8)
    syn34 = np.array(syn34).astype(np.int8)
    fault = np.array(fault).astype(np.int8)
    return syn12, syn34, fault, p, lu_avg, lu_std, data_size

class ioData:

    def __init__(self, syn12, syn34, fault):
        self.syn_12= syn12
        self.syn_34= syn34
        self.syn= np.concatenate((self.syn_12, self.syn_34), \
            axis= 1).reshape(-1, 2, 12)
        self.fault = fault
        self.fault_digit= np.dot(self.fault, [[8], [4], [2], [1]]) 
        self.fault_ind = y2indicator(self.fault_digit, 16)

def train(filename, param, graph):

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

    Syn12, Syn34, Fault, \
    p, lu_avg, lu_std, data_size = get_data(filename)
    output['res']['p']= p
    output['res']['lu avg']= lu_avg
    output['res']['lu std']= lu_std

    total_size= np.shape(Syn12)[0]
    test_size= int(test_fraction * total_size)
    error_scale= 1.0*total_size/data_size
    output['data']['fault scale']= error_scale

    train_data = ioData(Syn12[:-test_size,], Syn34[:-test_size,], \
        Fault[:-test_size,])
    test_data = ioData(Syn12[-test_size:,], Syn34[-test_size:,], \
        Fault[-test_size:,])

    N, num_inputs, size_inputs = train_data.syn.shape
    n_batches = N // batch_size

    output['data']['total data size']= total_size
    output['data']['test set size']= test_size
    output['opt']['batch size']= batch_size
    output['opt']['number of batches']= n_batches

    # TF IO placehoders
    num_hidden = 60
    num_outputs = 1
    num_classes= 16

    tf.reset_default_graph()

    In= tf.placeholder(tf.float32, [None, num_inputs, size_inputs])
    Out= tf.placeholder(tf.float32, [None, num_classes])

    LSTM = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    LSTMOut, _ = tf.nn.dynamic_rnn(LSTM, In, dtype=tf.float32)
    init_biases= np.zeros(num_classes)
    init_weights= np.random.randn(num_hidden, num_classes)
    biases= tf.Variable(init_biases.astype(np.float32))
    weights= tf.Variable(init_weights.astype(np.float32))
    OutIsh= tf.matmul(LSTMOut[:,-1,:], weights) + biases

    cost= tf.reduce_sum(\
          tf.nn.softmax_cross_entropy_with_logits(logits=OutIsh, labels=Out))

    # Choose an optimizer
    train = tf.train.RMSPropOptimizer(learning_rate, \
        decay=decay_rate, momentum=momentum_val).minimize(cost)

    # This is the predict of the network in the active mode
    predict_Out = tf.argmax(OutIsh, 1)

    costs = []
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        for i in range(num_iterations):

            # train all batches
            for j in range(n_batches):
                beg= j * batch_size
                end= j * batch_size + batch_size                
                Syn_batch = train_data.syn[beg : end,]
                Out_batch = train_data.fault_ind[beg : end,]
                session.run(train, feed_dict={In: Syn_batch, Out: Out_batch})

        prediction= session.run(predict_Out,feed_dict={In: test_data.syn})

        avg= error_rate(prediction, test_data.fault_digit)
        output['res']['nn avg'] = error_scale * avg
        output['res']['nn std'] = np.sqrt(0)

    return output

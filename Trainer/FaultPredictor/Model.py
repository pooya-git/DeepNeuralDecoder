import numpy as np
import tensorflow as tf
import tqdm
from time import time
import matplotlib
import matplotlib.pyplot as plt
import Networks as nn
from random import randint
from util import cyc_pick, vec_to_index, perp, y2indicator

class Model(object):
    
    def __init__(self, path, spec):
        self.spec = spec
        raw_data, p, lu_avg, lu_std, total_size = self.get_data(path)
        self.data_size = np.shape(raw_data[raw_data.keys()[0]])[0]
        self.init_syn(raw_data)
        self.init_rec(raw_data)
        self.init_log_1hot()
        self.p = p
        self.lu_avg = lu_avg
        self.lu_std = lu_std
        self.total_size = total_size
        self.error_scale= 1.0 * self.data_size / total_size

    def get_data(self, path):
        pass

    def init_data(self, raw_data):
        pass

    def init_log_1hot(self):

        self.log_1hot= {}
        for key in self.spec.err_keys:
            err = self.check_fault_after_correction(\
                (self.rec[key] + \
                self.lookup_correction_from_error(self.rec[key], key)) % 2, key)
            self.log_1hot[key]= y2indicator(err, 2).astype(np.int8)

    def syn_from_generators(self, err, key):

        return np.dot(err, self.spec.G[perp(key)].transpose()) % 2

    def pure_correction(self, syn, key):

        assert (np.shape(syn)[1] == self.spec.syn_size)
        return np.dot(syn, self.spec.T[key]) % 2

    def lookup_correction(self, syn, key):

        assert (np.shape(syn)[1] == self.spec.syn_size)
        index= vec_to_index(syn)
        return self.spec.correctionMat[key][index.transpose().tolist()]

    def pure_correction_from_error(self, err, key):

        syn= self.syn_from_generators(err, key)
        return self.pure_correction(syn, key)

    def lookup_correction_from_error(self, err, key):

        syn= self.syn_from_generators(err, key)
        return self.lookup_correction(syn, key)

    def check_fault_after_correction(self, err, key):

        return np.dot(err, self.spec.L[perp(key)].transpose()) % 2

    def check_logical_fault(self, err, key):

        correction = self.lookup_correction_from_error(err, key)
        err_final= (correction + err) % 2
        return self.check_fault_after_correction(err_final, key)

    def num_logical_fault(self, pred, t_beg):

        error_counter= 0.0
        for i in range(self.test_size):
            t_index= (i + t_beg) % self.data_size
            for key in self.spec.err_keys:
                if not 1 in self.syn[key][t_index]: pred[key][i]=0
                if (self.check_logical_fault(( \
                        pred[key][i] * self.spec.L[key] \
                        + self.rec[key][t_index]) % 2, key)):
                    error_counter+=1
                    break
        return error_counter/self.test_size

    def cost_function(self, param, x, y, predict, keep_rate):

        if param['type']=='FF':
            return nn.ff_cost(param, self.spec, x, y, predict)
        elif param['type']=='RNN':
            return nn.rnn_cost(param, self.spec, x, y, predict)
        elif param['type']=='W-LSTM':
            return nn.weighted_lstm(param, self.spec, x, y, predict)
        elif param['type']=='DeepLSTM':
            return nn.deep_lstm_cost(param, self.spec, x, y, predict, keep_rate)
        elif param['type']=='TwoDeepLSTM':
            return nn.two_deep_lstm_cost(\
                param, self.spec, x, y, predict, keep_rate)
        elif param['type']=='3DCNN':
            return nn.surface_conv3d_cost(\
                param, self.spec, x, y, predict, keep_rate)
        else:
            print('Neural network type not supported.')

    def train(self, param):

        verbose= param['usr']['verbose']
        batch_size= param['opt']['batch size']
        learning_rate= param['opt']['learning rate']
        num_iterations= param['opt']['iterations']
        momentum_val= param['opt']['momentum']
        decay_rate= param['opt']['decay']
        pointer= randint(0, self.data_size - 1) # self.test_size * trial
        t_beg= (self.train_size + pointer) % self.data_size

        tf.reset_default_graph()
        x, y, predict= {}, {}, {}
        for key in self.spec.err_keys:
            with tf.variable_scope(key):
                x[key] = tf.placeholder(tf.float32, [None,self.spec.input_size])
                y[key] = tf.placeholder(tf.float32, [None,2])
        keep_rate= tf.placeholder(tf.float32)
        cost= self.cost_function(param['nn'], x, y, predict, keep_rate)
        train = tf.train.RMSPropOptimizer(learning_rate, decay=decay_rate, \
            momentum=momentum_val).minimize(cost)
        init = tf.global_variables_initializer()

        costs= []
        prediction= {}

        with tf.Session() as session:
            session.run(init)

            for i in tqdm.tqdm(range(num_iterations)):
                for j in range(self.num_batches):
                    beg= (j * batch_size + pointer) % self.data_size
                    feed_dict={}
                    for key in self.spec.err_keys:
                        feed_dict[x[key]]= \
                            cyc_pick(self.syn[key], beg, batch_size)
                        feed_dict[y[key]]= \
                            cyc_pick(self.log_1hot[key], beg, batch_size)
                        feed_dict[keep_rate]= param['nn']['keep rate']
                    session.run(train, feed_dict)
                
                if verbose:
                    feed_dict={}
                    for key in self.spec.err_keys:
                        feed_dict[x[key]]= \
                            cyc_pick(self.syn[key], t_beg, self.test_size)
                        feed_dict[y[key]]= \
                            cyc_pick(self.log_1hot[key], t_beg, self.test_size)
                    feed_dict[keep_rate]= 1.0
                    test_cost = session.run(cost, feed_dict)
                    costs.append(test_cost)

            num_test_batches= param['data']['num test batch'] if \
                'num test batch' in param['data'].keys() else 1
            test_batch_size= self.test_size / num_test_batches
            for j in range(num_test_batches):
                beg= (j * test_batch_size + t_beg) % self.data_size
                if j==num_test_batches-1:
                    test_batch_size+= self.test_size % num_test_batches
                feed_dict={}
                for key in self.spec.err_keys:
                    feed_dict[x[key]]= \
                        cyc_pick(self.syn[key], beg, test_batch_size)
                feed_dict[keep_rate]= 1.0
                for key in self.spec.err_keys:
                    res= session.run(predict[key], feed_dict)
                    if key in prediction.keys():
                        prediction[key]= np.append(prediction[key], res, axis=0)
                    else:
                        prediction[key]= res

        if verbose:
            plt.plot(costs)
            plt.show()

        return prediction, t_beg

    def iso_cost_function(self, param, x, y, predict, keep_rate, key= None):

        if param['type']=='RNN':
            return nn.iso_rnn(param, self.spec, x, y, predict, key)
        elif param['type']=='Conv3d':
            return nn.iso_conv3d(param, self.spec, x, y, predict, key)
        else:
            print('Neural network type not supported.')

    def iso_train(self, param):

        verbose= param['usr']['verbose']
        batch_size= param['opt']['batch size']
        learning_rate= param['opt']['learning rate']
        num_iterations= param['opt']['iterations']
        momentum_val= param['opt']['momentum']
        decay_rate= param['opt']['decay']
        pointer= randint(0, self.data_size - 1) # self.test_size * trial
        t_beg= (self.train_size + pointer) % self.data_size

        tf.reset_default_graph()
        x, y, predict, cost, train= {}, {}, {}, {}, {}
        for key in self.spec.err_keys:
            with tf.variable_scope(key):
                x[key] = tf.placeholder(tf.float32, [None,self.spec.input_size])
                y[key] = tf.placeholder(tf.float32, [None,2])
                keep_rate= tf.placeholder(tf.float32)
                cost[key]= self.iso_cost_function(\
                    param['nn'], x[key], y[key], predict, keep_rate, key)
                train[key] = tf.train.RMSPropOptimizer(\
                    learning_rate, decay=decay_rate, \
                    momentum=momentum_val).minimize(cost[key])
        init = tf.global_variables_initializer()

        costs= []
        prediction= {}

        with tf.Session() as session:
            session.run(init)

            for i in tqdm.tqdm(range(num_iterations)):
                for j in range(self.num_batches):
                    beg= (j * batch_size + pointer) % self.data_size
                    for key in self.spec.err_keys:
                        feed_dict={}
                        feed_dict[x[key]]= \
                            cyc_pick(self.syn[key], beg, batch_size)
                        feed_dict[y[key]]= \
                            cyc_pick(self.log_1hot[key], beg, batch_size)
                        feed_dict[keep_rate]= param['nn']['keep rate']
                        session.run(train[key], feed_dict)
                
                if verbose:
                    test_cost= []
                    for key in self.spec.err_keys:
                        feed_dict={}
                        feed_dict[x[key]]= \
                            cyc_pick(self.syn[key], t_beg, self.test_size)
                        feed_dict[y[key]]= \
                            cyc_pick(self.log_1hot[key], t_beg, self.test_size)
                        feed_dict[keep_rate]= 1.0
                        test_cost.append(session.run(cost[key], feed_dict))
                    costs.append(test_cost)

            start_time= time()
            feed_dict={}
            for key in self.spec.err_keys:
                feed_dict[x[key]]= \
                    cyc_pick(self.syn[key], t_beg, self.test_size)
                feed_dict[keep_rate]= 1.0
                prediction[key] = session.run(predict[key], feed_dict)

        if verbose:
            plt.plot(costs)
            plt.show()

        return prediction, t_beg

    def mixed_cost_function(self, param, x, y, predict, keep_rate, pair):

        if param['type']=='MixedFF':
            return nn.mixed_ff(param, self.spec, x, y, predict, pair)
        elif param['type']=='MixedConv3d':
            return nn.mixed_conv3d(param, self.spec, x, y, predict, pair)
        elif param['type']=='MixedRNN':
            return nn.mixed_rnn(param, self.spec, x, y, predict, pair)
        else:
            print('Neural network type not supported.')

    def mixed_train(self, param):

        verbose= param['usr']['verbose']
        batch_size= param['opt']['batch size']
        learning_rate= param['opt']['learning rate']
        num_iterations= param['opt']['iterations']
        momentum_val= param['opt']['momentum']
        decay_rate= param['opt']['decay']
        pointer= randint(0, self.data_size - 1) # self.test_size * trial
        t_beg= (self.train_size + pointer) % self.data_size

        tf.reset_default_graph()
        x, y, predict, cost, train= {}, {}, {}, {}, {}
        for pair in self.spec.perp_keys:
            with tf.variable_scope(''.join(pair)):
                for ind in pair:
                    x[ind] = tf.placeholder(tf.float32, [None, self.spec.input_size])
                    y[ind] = tf.placeholder(tf.float32, [None, 2])
                keep_rate= tf.placeholder(tf.float32)
                cost[pair]= self.mixed_cost_function(\
                    param['nn'], x, y, predict, keep_rate, pair)
                train[pair] = tf.train.RMSPropOptimizer(\
                    learning_rate, decay=decay_rate, \
                    momentum=momentum_val).minimize(cost[pair])
        init = tf.global_variables_initializer()

        costs= []
        prediction= {}

        with tf.Session() as session:
            session.run(init)

            for i in tqdm.tqdm(range(num_iterations)):
                for j in range(self.num_batches):
                    beg= (j * batch_size + pointer) % self.data_size
                    for pair in self.spec.perp_keys:
                        feed_dict={}
                        for ind in pair:
                            feed_dict[x[ind]]= \
                                cyc_pick(self.syn[ind], beg, batch_size)
                            feed_dict[y[ind]]= \
                                cyc_pick(self.log_1hot[ind], beg, batch_size)
                        feed_dict[keep_rate]= param['nn']['keep rate']
                        session.run(train[pair], feed_dict)
                
                if verbose:
                    test_cost= []
                    for pair in self.spec.perp_keys:
                        feed_dict={}
                        for ind in pair:
                            feed_dict[x[ind]]= \
                                cyc_pick(self.syn[ind], t_beg, self.test_size)
                            feed_dict[y[ind]]= \
                            cyc_pick(self.log_1hot[ind], t_beg, self.test_size)
                        feed_dict[keep_rate]= 1.0
                        test_cost.append(session.run(cost[pair], feed_dict))
                    costs.append(test_cost)

            start_time= time()
            for pair in self.spec.perp_keys:
                feed_dict={}
                feed_dict[keep_rate]= 1.0
                for ind in pair:
                    feed_dict[x[ind]]= \
                        cyc_pick(self.syn[ind], t_beg, self.test_size)
                for ind in pair:
                    prediction[ind] = session.run(predict[ind], feed_dict)

        if verbose:
            plt.plot(costs)
            plt.show()

        return prediction, t_beg

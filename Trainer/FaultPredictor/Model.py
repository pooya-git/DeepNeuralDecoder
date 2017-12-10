import numpy as np
import tensorflow as tf
import tqdm
from time import time
import matplotlib
import matplotlib.pyplot as plt
import Networks as nn
from random import randint
from util import cyc_pick

class Model(object):
    
    def __init__(self, path, spec):
        self.spec = spec
        raw_data, p, lu_avg, lu_std, total_size = self.get_data(path)
        self.data_size = np.shape(raw_data[raw_data.keys()[0]])[0]
        self.init_data(raw_data)
        self.p = p
        self.lu_avg = lu_avg
        self.lu_std = lu_std
        self.total_size = total_size
        self.error_scale= 1.0 * self.data_size / total_size

    def get_data(self, path):
        pass

    def init_data(self, raw_data):
        pass

    def num_logical_fault(self, prediction, test_beg):
        pass

    def cost_function(self, param, x, y, predict, keep_rate):

        if param['type']=='FF':
            return nn.ff_cost(param, self.spec, x, y, predict)
        elif param['type']=='Conv3d':
            return nn.surface_conv3d_cost(param, self.spec, x, y, predict)
        elif param['type']=='RNN':
            return nn.rnn_cost(param, self.spec, x, y, predict)
        elif param['type']=='W-LSTM':
            return nn.weighted_lstm(param, self.spec, x, y, predict)
        elif param['type']=='DeepLSTM':
            return nn.deep_lstm_cost(param, self.spec, x, y, predict, keep_rate)
        elif param['type']=='TwoDeepLSTM':
            return nn.two_deep_lstm_cost(param,self.spec,x,y,predict, keep_rate)
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

            start_time= time()
            feed_dict={}
            for key in self.spec.err_keys:
                feed_dict[x[key]]= \
                    cyc_pick(self.syn[key], t_beg, self.test_size)
            feed_dict[keep_rate]= 1.0
            for key in self.spec.err_keys:
                prediction[key] = session.run(predict[key], feed_dict)

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

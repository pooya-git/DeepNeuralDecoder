import numpy as np
import tensorflow as tf
import tqdm
from time import time
import matplotlib
import matplotlib.pyplot as plt
import Networks as nn

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

        if param['type']=='LogReg':
            return nn.logistic_regression(param, self.spec, x, y, predict)
        elif param['type']=='FF':
            return nn.ff_cost(param, self.spec, x, y, predict)
        elif param['type']=='XFF':
            return nn.cross_ff_cost(param, self.spec, x, y, predict)
        elif param['type']=='Conv3d':
            return nn.surface_conv3d_cost(param, self.spec, x, y, predict)
        elif param['type']=='LSTM':
            return nn.lstm_cost(param, self.spec, x, y, predict)
        elif param['type']=='W-LSTM':
            return nn.weighted_lstm(param, self.spec, x, y, predict)
        elif param['type']=='DeepLSTM':
            return nn.deep_lstm_cost(param, self.spec, x, y, predict, keep_rate)
        elif param['type']=='TwoDeepLSTM':
            return nn.two_deep_lstm_cost(param, self.spec, x, y, predict, keep_rate)
        else:
            print('Neural network type not supportd.')

    def train(self, param, trial= 0):

        verbose= param['usr']['verbose']
        batch_size= param['opt']['batch size']
        learning_rate= param['opt']['learning rate']
        num_iterations= param['opt']['iterations']
        momentum_val= param['opt']['momentum']
        decay_rate= param['opt']['decay']
        pointer= self.test_size * trial
        t_beg= (self.train_size + pointer) % self.data_size
        t_end= (self.train_size + self.test_size + pointer) % self.data_size
        if not t_end: t_end = None

        tf.reset_default_graph()
        x, y, predict= {}, {}, {}
        for key in self.spec.err_keys:
            with tf.variable_scope(key):
                x[key] = tf.placeholder(tf.float32, [None, self.spec.input_size])
                y[key] = tf.placeholder(tf.float32, [None, 2])
        keep_rate= tf.placeholder(tf.float32)
        cost= self.cost_function(param['nn'], x, y, predict, keep_rate)
        train = tf.train.RMSPropOptimizer(learning_rate, decay=decay_rate, \
            momentum=momentum_val).minimize(cost)
        init = tf.global_variables_initializer()

        costs= []
        prediction= {}

        # inputs= {}
        # for key in self.spec.err_keys:
        #     # mu = self.syn[key].mean(axis=0)
        #     # std = self.syn[key].std(axis=0)
        #     # inputs[key]= (self.syn[key] - mu) / std
        #     inputs[key]= self.syn[key]

        with tf.Session() as session:
            session.run(init)

            for i in tqdm.tqdm(range(num_iterations)):
                for j in range(self.num_batches):
                    beg= (j*batch_size + pointer) % self.data_size
                    end= (j*batch_size + batch_size + pointer) % self.data_size
                    if not end: end = None
                    feed_dict={}
                    for key in self.spec.err_keys:
                        if (beg < end):
                            feed_dict[x[key]]= self.syn[key][beg:end]
                            feed_dict[y[key]]= self.log_1hot[key][beg:end]
                        else:
                            feed_dict[x[key]]= np.concatenate(\
                                (self.syn[key][beg:],\
                                 self.syn[key][:end]), axis=0)
                            feed_dict[y[key]]= np.concatenate(\
                                (self.log_1hot[key][beg:],\
                                 self.log_1hot[key][:end]), axis=0)
                        feed_dict[keep_rate]= param['nn']['keep rate']
                    session.run(train, feed_dict)
                
                if verbose:
                    feed_dict={}
                    for key in self.spec.err_keys:
                        feed_dict[x[key]]= self.syn[key][t_beg:t_end]
                        feed_dict[y[key]]= self.log_1hot[key][t_beg:t_end]
                    feed_dict[keep_rate]= 1.0
                    test_cost = session.run(cost, feed_dict)
                    costs.append(test_cost)

            start_time= time()
            feed_dict={}
            for key in self.spec.err_keys:
                feed_dict[x[key]]= self.syn[key][t_beg:t_end]
            feed_dict[keep_rate]= 1.0
            for key in self.spec.err_keys:
                prediction[key] = session.run(predict[key], feed_dict)

        if verbose:
            plt.plot(costs)
            plt.show()

        return prediction, t_beg

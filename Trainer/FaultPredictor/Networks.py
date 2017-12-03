# ------------------------------------------------------------------------------
# 
#    Tensorflow trainer functions.
#
#    Copyright (C) 2017 Pooya Ronagh
# 
# ------------------------------------------------------------------------------

import tensorflow as tf
from util import perp

def cross_ff_cost(param, spec, x, y, predict):

    num_hiddens= param['num hidden'] 
    W_std= param['W std'] 
    b_std= param['b std']
    W11, b11, W12, b12, W21, b21, W22, b22 = {}, {}, {}, {}, {}, {}, {}, {}
    hidden, logits, loss = {}, {}, {}

    for key in spec.err_keys:
        with tf.variable_scope(key):
            W11[key]= tf.Variable(tf.random_normal(\
                [spec.input_size, num_hiddens], stddev=W_std))
            b11[key]= tf.Variable(tf.random_normal([num_hiddens], stddev=b_std))
            W12[key]= tf.Variable(tf.random_normal(\
                [spec.input_size, num_hiddens], stddev=W_std))
            b12[key]= tf.Variable(tf.random_normal([num_hiddens], stddev=b_std))
            hidden[key]= tf.nn.relu(tf.matmul(x[key], W11[key]) + b11[key]\
                + tf.matmul(x[key], W12[key]) + b12[key])
    for key in spec.err_keys:
        with tf.variable_scope(key):
            W21[key]= tf.Variable(tf.random_normal(\
                [num_hiddens, spec.num_labels], stddev=W_std))
            b21[key]= tf.Variable(tf.random_normal(\
                [spec.num_labels], stddev=b_std))
            W22[key]= tf.Variable(tf.random_normal(\
                [num_hiddens, spec.num_labels], stddev=W_std))
            b22[key]= tf.Variable(tf.random_normal(\
                [spec.num_labels], stddev=b_std))
    for key in spec.err_keys:
        with tf.variable_scope(key):
            logits[key]= tf.nn.relu(tf.matmul(hidden[key], W21[key]) +b21[key]\
                + tf.matmul(hidden[perp(key)], W22[key]) +b22[key])
            loss[key]= tf.nn.softmax_cross_entropy_with_logits(\
                logits=logits[key], labels=y[key])
            predict[key]= tf.argmax(logits[key], 1)
    return tf.reduce_sum(sum(loss[key] for key in spec.err_keys))

def surface_conv3d_cost(param, spec, x, y, predict):

    num_hiddens= param['num hidden']
    num_filters= param['num filters']
    kernel_size= param['kernel size']
    flat_size= (spec.d - 1) * (spec.syn_w - 1) * (spec.syn_h - 1) * num_filters
    W_std= param['W std'] 
    b_std= param['b std']
    conv_input, conv, pool, pool_flat= {}, {}, {}, {}
    W1, b1, W2, b2 = {}, {}, {}, {}
    hidden, logits, loss = {}, {}, {}

    for key in spec.err_keys:
        with tf.variable_scope(key):
            conv_input[key] = tf.reshape(\
                x[key],[-1, spec.d, spec.syn_w, spec.syn_h, 1])
            conv[key] = tf.layers.conv3d(\
                conv_input[key], filters= num_filters,\
                kernel_size= kernel_size,\
                padding= 'same', activation=tf.nn.relu)
            pool[key] = tf.layers.max_pooling3d(conv[key],\
                pool_size=[2, 2, 2], strides=1)
            pool_flat[key]= tf.reshape(pool[key], [-1, flat_size])
            # W1[key]= tf.Variable(\
            #     tf.random_normal([flat_size, num_hiddens], stddev=W_std))
            # b1[key]= tf.Variable(tf.random_normal([num_hiddens], stddev=b_std))
            # hidden[key]= tf.nn.relu(tf.matmul(pool_flat[key], W1[key])+ b1[key])
            W2[key]= tf.Variable(\
                tf.random_normal([flat_size, spec.num_labels], stddev=W_std))
            b2[key]= tf.Variable(\
                tf.random_normal([spec.num_labels], stddev=b_std))
            logits[key]= (tf.matmul(pool_flat[key], W2[key]) +b2[key])
            loss[key]= tf.nn.softmax_cross_entropy_with_logits(\
                logits=logits[key], labels=y[key])
            predict[key]= tf.argmax(logits[key], 1)    
    return tf.reduce_sum(sum(loss[key] for key in spec.err_keys))

def ff_cost(param, spec, x, y, predict):

    num_hiddens= [spec.input_size] + param['num hidden'] + [spec.num_labels]
    W_std= param['W std'] 
    b_std= param['b std']
    layer, logits, loss = {}, {}, {}

    for key in spec.err_keys:
        layer[key]= []
        layer[key].append(x[key])
        with tf.variable_scope(key):
            for l in range(len(num_hiddens)-1):
                W= tf.Variable(tf.random_normal(\
                    [num_hiddens[l], num_hiddens[l+1]], stddev=W_std))
                b= tf.Variable(tf.random_normal(\
                    [num_hiddens[l+1]], stddev=b_std))
                layer[key].append(tf.nn.relu(tf.matmul(layer[key][-1], W) + b))
            loss[key]= tf.nn.softmax_cross_entropy_with_logits(\
                logits=layer[key][-1], labels=y[key])
            predict[key]= tf.argmax(layer[key][-1], 1)
    return tf.reduce_sum(sum(loss[key] for key in spec.err_keys))

def logistic_regression(param, spec, x, y, predict):

    num_hiddens= param['num hidden'] 
    W_std= param['W std'] 
    b_std= param['b std']
    W1, b1= {}, {}
    logits, loss = {}, {}

    for key in spec.err_keys:
        with tf.variable_scope(key):
            W1[key]= tf.Variable(tf.random_normal(\
                [spec.input_size, spec.num_labels], stddev=W_std))
            b1[key]= tf.Variable(tf.random_normal(\
                [spec.num_labels], stddev=b_std))
            logits[key]= tf.nn.sigmoid(tf.matmul(x[key], W1[key]) + b1[key])
            loss[key]= tf.nn.softmax_cross_entropy_with_logits(\
                logits=logits[key], labels=y[key])
            predict[key]= tf.argmax(logits[key], 1)
    return tf.reduce_sum(sum(loss[key] for key in spec.err_keys))

def weighted_lstm(param, spec, x, y, predict):

    num_hiddens= param['num hidden']
    W_std= param['W std'] 
    b_std= param['b std']
    W, b = {}, {}
    lstmIn, lstm, lstmOut= {}, {}, {}
    logits, loss = {}, {}
    weights= {}

    for key in spec.err_keys:
        with tf.variable_scope(key):
            lstmIn[key]= tf.reshape(x[key], \
                [-1, spec.num_epochs, spec.lstm_input_size])
            lstm[key] = tf.contrib.rnn.LSTMCell(num_hiddens)
            lstmOut[key], _ = tf.nn.dynamic_rnn(\
                lstm[key], lstmIn[key], dtype=tf.float32)
            W[key]= tf.Variable(\
                tf.random_normal([num_hiddens, spec.num_labels], stddev=W_std))
            b[key]= tf.Variable(\
                tf.random_normal([spec.num_labels], stddev=b_std))
            logits[key]= tf.matmul(lstmOut[key][:, -1, :], W[key]) + b[key]            
            loss[key]= tf.nn.weighted_cross_entropy_with_logits(\
                logits=logits[key], targets=y[key], pos_weight= 1.0/168)
            predict[key]= tf.argmax(logits[key], 1)
    return tf.reduce_sum(sum(loss[key] for key in spec.err_keys))

def lstm_cost(param, spec, x, y, predict):

    num_hiddens= param['num hidden']
    W_std= param['W std'] 
    b_std= param['b std']
    W, b = {}, {}
    lstmIn, lstm, lstmOut= {}, {}, {}
    logits, loss = {}, {}

    for key in spec.err_keys:
        with tf.variable_scope(key):
            lstmIn[key]= tf.reshape(x[key], \
                [-1, spec.num_epochs, spec.lstm_input_size])
            lstm[key] = tf.contrib.rnn.LSTMCell(num_hiddens)
            lstmOut[key], _ = tf.nn.dynamic_rnn(\
                lstm[key], lstmIn[key], dtype=tf.float32)
            W[key]= tf.Variable(\
                tf.random_normal([num_hiddens, spec.num_labels], stddev=W_std))
            b[key]= tf.Variable(\
                tf.random_normal([spec.num_labels], stddev=b_std))
            logits[key]= tf.matmul(lstmOut[key][:, -1, :], W[key]) + b[key]
            loss[key]= tf.nn.softmax_cross_entropy_with_logits(\
                logits=logits[key], labels=y[key])
            predict[key]= tf.argmax(logits[key], 1)
    return tf.reduce_sum(sum(loss[key] for key in spec.err_keys))

def deep_lstm_cost(param, spec, x, y, predict, drop_rate):

    num_hiddens= param['num hidden']
    W_std= param['W std'] 
    b_std= param['b std']
    W1, b1, W2, b2 = {}, {}, {}, {}
    lstmIn, lstm, lstmOut= {}, {}, {}
    logits, loss = {}, {}
    fc_layer = {}
    out= {}

    for key in spec.err_keys:
        with tf.variable_scope(key):
            lstmIn[key]= tf.reshape(x[key], \
                [-1, spec.num_epochs, spec.lstm_input_size])
            lstm[key] = tf.contrib.rnn.LSTMCell(num_hiddens)
            lstmOut[key], _ = tf.nn.dynamic_rnn(\
                lstm[key], lstmIn[key], dtype=tf.float32)
            W1[key]= tf.Variable(\
                tf.random_normal([num_hiddens, num_hiddens], stddev=W_std))
            b1[key]= tf.Variable(\
                tf.random_normal([num_hiddens], stddev=b_std))
            fc_layer[key]= tf.nn.relu(tf.matmul(lstmOut[key][:, -1, :], W1[key]) + b1[key])
            out[key] = tf.nn.dropout(fc_layer[key], drop_rate)
            W2[key]= tf.Variable(\
                tf.random_normal([num_hiddens, spec.num_labels], stddev=W_std))
            b2[key]= tf.Variable(\
                tf.random_normal([spec.num_labels], stddev=b_std))
            logits[key]= tf.matmul(out[key], W2[key]) + b2[key]
            loss[key]= tf.nn.softmax_cross_entropy_with_logits(\
                logits=logits[key], labels=y[key])
            predict[key]= tf.argmax(logits[key], 1)
    return tf.reduce_sum(sum(loss[key] for key in spec.err_keys))

def two_deep_lstm_cost(param, spec, x, y, predict, drop_rate):

    num_hiddens= param['num hidden']
    W_std= param['W std'] 
    b_std= param['b std']
    W1, b1, W2, b2, W3, b3 = {}, {}, {}, {}, {}, {}
    lstmIn, lstm, lstmOut= {}, {}, {}
    logits, loss = {}, {}
    fc_layer1, fc_layer2 = {}, {}
    out= {}

    for key in spec.err_keys:
        with tf.variable_scope(key):
            lstmIn[key]= tf.reshape(x[key], \
                [-1, spec.num_epochs, spec.lstm_input_size])
            lstm[key] = tf.contrib.rnn.LSTMCell(num_hiddens)
            lstmOut[key], _ = tf.nn.dynamic_rnn(\
                lstm[key], lstmIn[key], dtype=tf.float32)
            W1[key]= tf.Variable(\
                tf.random_normal([num_hiddens, num_hiddens], stddev=W_std))
            b1[key]= tf.Variable(\
                tf.random_normal([num_hiddens], stddev=b_std))
            fc_layer1[key]= tf.nn.relu(tf.matmul(lstmOut[key][:, -1, :], W1[key]) + b1[key])
            
            W2[key]= tf.Variable(\
                tf.random_normal([num_hiddens, num_hiddens], stddev=W_std))
            b2[key]= tf.Variable(\
                tf.random_normal([num_hiddens], stddev=b_std))
            fc_layer2[key]= tf.nn.relu(tf.matmul(fc_layer1[key], W2[key]) + b2[key])

            out[key] = tf.nn.dropout(fc_layer2[key], drop_rate)

            W3[key]= tf.Variable(\
                tf.random_normal([num_hiddens, spec.num_labels], stddev=W_std))
            b3[key]= tf.Variable(\
                tf.random_normal([spec.num_labels], stddev=b_std))
            logits[key]= tf.matmul(out[key], W3[key]) + b3[key]
            loss[key]= tf.nn.softmax_cross_entropy_with_logits(\
                logits=logits[key], labels=y[key])
            predict[key]= tf.argmax(logits[key], 1)
    return tf.reduce_sum(sum(loss[key] for key in spec.err_keys))
# ------------------------------------------------------------------------------
# 
#    Tensorflow trainer functions.
#
#    Copyright (C) 2017 Pooya Ronagh
# 
# ------------------------------------------------------------------------------

import tensorflow as tf
from util import perp
import CustomLSTM

def cross_ff_cost(param, spec, x, y, predict):

    num_hiddens= param['num hidden'][0] 
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

    num_hiddens= param['num hidden'][0]
    num_filters= param['num filters']
    kernel_size= param['kernel size']
    pad_size= param['padding size']
    flat_size= (spec.d) * \
               (2 * pad_size + spec.syn_w) * \
               (2 * pad_size + spec.syn_h) * num_filters
    W_std= param['W std'] 
    b_std= param['b std']
    conv_input, padded_input, conv, pool, pool_flat= {}, {}, {}, {}, {}
    hidden, logits, loss = {}, {}, {}

    for key in spec.err_keys:
        with tf.variable_scope(key):
            conv_input[key] = tf.reshape(\
                x[key],[-1, spec.d, spec.syn_w, spec.syn_h, 1])
            padded_input[key] = tf.pad(conv_input[key], \
                tf.constant([[0,0],[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]]),\
                'SYMMETRIC')
            conv[key] = tf.layers.conv3d(\
                padded_input[key], filters= num_filters,\
                kernel_size= kernel_size,\
                padding= 'same', activation=tf.nn.relu)
            # pool[key] = tf.layers.max_pooling3d(conv[key],\
            #     pool_size=2, strides=1)
            pool_flat[key]= tf.reshape(conv[key], [-1, flat_size])
            # W1[key]= tf.Variable(\
            #     tf.random_normal([flat_size, num_hiddens], stddev=W_std))
            # b1[key]= tf.Variable(tf.random_normal([num_hiddens], stddev=b_std))
            # hidden[key]= tf.nn.relu(tf.matmul(pool_flat[key], W1[key])+ b1[key])
            W= tf.Variable(\
                tf.random_normal([flat_size, spec.num_labels], stddev=W_std))
            b= tf.Variable(\
                tf.random_normal([spec.num_labels], stddev=b_std))
            logits[key]= (tf.matmul(pool_flat[key], W) +b)
            loss[key]= tf.nn.softmax_cross_entropy_with_logits(\
                logits=logits[key], labels=y[key])
            predict[key]= tf.argmax(logits[key], 1)    
    return tf.reduce_sum(sum(loss[key] for key in spec.err_keys))

def ff_cost(param, spec, x, y, predict):

    num_hiddens= [spec.input_size] + param['num hidden'] + [spec.num_labels]
    activations= []
    for i in range(len(param['activations'])):
        if param['activations'][i]=='relu':
            activations.append(tf.nn.relu)
        elif param['activations'][i]=='sigmoid':
            activations.append(tf.nn.sigmoid)
        elif param['activations'][i]=='id':
            activations.append(tf.identity)
        elif param['activations'][i]=='tanh':
            activations.append(tf.tanh)
        else:
            raise Exception('Activation function not recognized.')
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
                layer[key].append(\
                    activations[l](tf.matmul(layer[key][-1], W) + b))
            loss[key]= tf.nn.softmax_cross_entropy_with_logits(\
                logits=layer[key][-1], labels=y[key])
            predict[key]= tf.argmax(layer[key][-1], 1)
    return tf.reduce_sum(sum(loss[key] for key in spec.err_keys))

def weighted_lstm(param, spec, x, y, predict):

    num_hiddens= param['num hidden'][0]
    W_std= param['W std'] 
    b_std= param['b std']
    pos_weight= param['positive weight']
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
                logits=logits[key], targets=y[key], pos_weight= pos_weight)
            predict[key]= tf.argmax(logits[key], 1)
    return tf.reduce_sum(sum(loss[key] for key in spec.err_keys))

def iso_rnn(param, spec, x, y, predict, key):

    num_hiddens= param['num hidden'][0]
    W_std= param['W std'] 
    b_std= param['b std']
    if param['unit type']=='LSTMCell':
        lstm_cell= tf.contrib.rnn.LSTMCell
    elif param['unit type']=='GRUCell':
        lstm_cell= tf.contrib.rnn.GRUCell

    with tf.variable_scope(key):
        lstmIn= tf.reshape(x, [-1, spec.num_epochs, spec.lstm_input_size])
        lstm = lstm_cell(num_hiddens)
        lstmOut, _ = tf.nn.dynamic_rnn(lstm, lstmIn, dtype=tf.float32)
        W= tf.Variable(\
            tf.random_normal([num_hiddens, spec.num_labels], stddev=W_std))
        b= tf.Variable(tf.random_normal([spec.num_labels], stddev=b_std))
        logits= tf.matmul(lstmOut[:, -1, :], W) + b
        loss= tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        predict[key]= tf.argmax(logits, 1)
    return tf.reduce_sum(loss)

def iso_conv3d(param, spec, x, y, predict, key):

    num_hiddens= param['num hidden'][0]
    num_filters= param['num filters']
    kernel_size= param['kernel size']
    pad_size= param['padding size']
    flat_size= (spec.d) * \
               (2 * pad_size + spec.syn_w) * \
               (2 * pad_size + spec.syn_h) * num_filters
    W_std= param['W std'] 
    b_std= param['b std']

    with tf.variable_scope(key):
        conv_input = tf.reshape(\
            x,[-1, spec.d, spec.syn_w, spec.syn_h, 1])
        padded_input = tf.pad(conv_input, tf.constant(\
            [[0,0], [0,0], [pad_size,pad_size], [pad_size,pad_size], [0,0]]),\
            'SYMMETRIC')
        conv = tf.layers.conv3d(padded_input, filters= num_filters,\
            kernel_size= kernel_size, padding= 'same', activation=tf.nn.relu)
        pool_flat= tf.reshape(conv, [-1, flat_size])
        W= tf.Variable(\
            tf.random_normal([flat_size, spec.num_labels], stddev=W_std))
        b= tf.Variable(tf.random_normal([spec.num_labels], stddev=b_std))
        logits= (tf.matmul(pool_flat, W) + b)
        loss= tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        predict[key]= tf.argmax(logits, 1)    
    return tf.reduce_sum(loss)

def rnn_cost(param, spec, x, y, predict):

    num_hiddens= param['num hidden'][0]
    use_peepholes= param['peepholes']
    W_std= param['W std'] 
    b_std= param['b std']
    if param['unit type']=='LSTM':
        lstm_cell= tf.contrib.rnn.LSTMCell
    elif param['unit type']=='GRU':
        lstm_cell= tf.contrib.rnn.GRUCell
    elif param['unit type']=='Custom':
        lstm_cell= CustomLSTM.CustomLSTMCell
    else:
        raise Exception('RNN cell not recognized.')
    lstmIn, lstm, lstmOut= {}, {}, {}
    logits, loss = {}, {}

    for key in spec.err_keys:
        with tf.variable_scope(key):
            lstmIn[key]= tf.reshape(x[key], \
                [-1, spec.num_epochs, spec.lstm_input_size])
            lstm[key] = lstm_cell(num_hiddens)
            lstmOut[key], _ = tf.nn.dynamic_rnn(\
                lstm[key], lstmIn[key], dtype=tf.float32)
            W= tf.Variable(\
                tf.random_normal([num_hiddens, spec.num_labels], stddev=W_std))
            b= tf.Variable(\
                tf.random_normal([spec.num_labels], stddev=b_std))
            logits[key]= tf.matmul(lstmOut[key][:, -1, :], W) + b
            loss[key]= tf.nn.softmax_cross_entropy_with_logits(\
                logits=logits[key], labels=y[key])
            predict[key]= tf.argmax(logits[key], 1)
    return tf.reduce_sum(sum(loss[key] for key in spec.err_keys))

def deep_lstm_cost(param, spec, x, y, predict, drop_rate):

    num_hiddens= param['num hidden'][0]
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

    num_hiddens= param['num hidden'][0]
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

def mixed_conv3d(param, spec, x, y, predict):

    num_hiddens= param['num hidden'][0]
    num_filters= param['num filters']
    kernel_size= param['kernel size']
    pad_size= param['padding size']
    flat_size= (spec.d) * \
               (2 * pad_size + spec.syn_w) * \
               (2 * pad_size + spec.syn_h) * num_filters
    W_std= param['W std'] 
    b_std= param['b std']
    conv_input, padded_input= {}, {}

    mixed_y= tf.one_hot(tf.argmax(y['X'], 1) + 2*tf.argmax(y['Z'], 1), 4)

    for key in spec.err_keys:
        with tf.variable_scope(key):
            conv_input[key] = tf.reshape(\
                x[key],[-1, spec.d, spec.syn_w, spec.syn_h, 1])
            padded_input[key] = tf.pad(conv_input[key], \
                tf.constant([[0,0],[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]]),\
                'SYMMETRIC')
    mixed_input = tf.concat([padded_input['X'], padded_input['Z']], 4)    
    mixed_conv = tf.layers.conv3d(\
        mixed_input, filters= num_filters,\
        kernel_size= kernel_size,\
        padding= 'same', activation=tf.nn.relu)
    # pool[key] = tf.layers.max_pooling3d(conv[key],\
    #     pool_size=2, strides=1)
    mixed_pool_flat= tf.reshape(mixed_conv, [-1, flat_size])
            # W1[key]= tf.Variable(\
            #     tf.random_normal([flat_size, num_hiddens], stddev=W_std))
            # b1[key]= tf.Variable(tf.random_normal([num_hiddens], stddev=b_std))
            # hidden[key]= tf.nn.relu(tf.matmul(pool_flat[key], W1[key])+ b1[key])
    W= tf.Variable(\
        tf.random_normal([flat_size, 2*spec.num_labels], stddev=W_std))
    b= tf.Variable(tf.random_normal([2*spec.num_labels], stddev=b_std))
    mixed_logits= (tf.matmul(mixed_pool_flat, W) +b)
    mixed_loss= tf.nn.softmax_cross_entropy_with_logits(\
        logits=mixed_logits, labels=mixed_y)
    mixed_predict= tf.argmax(mixed_logits, 1)

    predict['X']= mixed_predict % 2
    predict['Z']= mixed_predict // 2
    
    return tf.reduce_sum(mixed_loss)

def mixed_ff(param, spec, x, y, predict, perp_keys):

    num_hiddens= [2*spec.input_size] + param['num hidden'] + [2*spec.num_labels]
    activations= []
    for i in range(len(param['activations'])):
        if param['activations'][i]=='relu':
            activations.append(tf.nn.relu)
        elif param['activations'][i]=='sigmoid':
            activations.append(tf.nn.sigmoid)
        elif param['activations'][i]=='id':
            activations.append(tf.identity)
        elif param['activations'][i]=='tanh':
            activations.append(tf.tanh)
        else:
            raise Exception('Activation function not recognized.')
    W_std= param['W std'] 
    b_std= param['b std']

    layer= []
    mixed_y= tf.one_hot(\
        tf.argmax(y[perp_keys[0]], 1) + 2 * tf.argmax(y[perp_keys[1]], 1), 4)

    layer.append(tf.concat([x[key] for key in perp_keys], 1) )
    for l in range(len(num_hiddens)-1):
        W= tf.Variable(tf.random_normal(\
            [num_hiddens[l], num_hiddens[l+1]], stddev=W_std))
        b= tf.Variable(tf.random_normal([num_hiddens[l+1]], stddev=b_std))
        layer.append(activations[l](tf.matmul(layer[-1], W) + b))
    loss= tf.nn.softmax_cross_entropy_with_logits(\
        logits=layer[-1], labels=mixed_y)
    mixed_predict= tf.argmax(layer[-1], 1)
    predict[perp_keys[0]]= mixed_predict % 2
    predict[perp_keys[1]]= mixed_predict // 2

    return tf.reduce_sum(loss)


def mixed_rnn(param, spec, x, y, predict, perp_keys):

    num_hiddens= param['num hidden'][0]
    W_std= param['W std'] 
    b_std= param['b std']
    if param['unit type']=='LSTMCell':
        lstm_cell= tf.contrib.rnn.LSTMCell
    elif param['unit type']=='GRUCell':
        lstm_cell= tf.contrib.rnn.GRUCell

    mixed_y= tf.one_hot(\
        tf.argmax(y[perp_keys[0]], 1) + 2 * tf.argmax(y[perp_keys[1]], 1), 4)
    mixed_x= tf.stack([x[perp_keys[0]], x[perp_keys[1]]], axis=1)
    lstmIn= tf.reshape(mixed_x, [-1, spec.num_epochs, 2 * spec.lstm_input_size])
    lstm = lstm_cell(num_hiddens)
    lstmOut, _ = tf.nn.dynamic_rnn(lstm, lstmIn, dtype=tf.float32)
    W= tf.Variable(\
        tf.random_normal([num_hiddens, 2 * spec.num_labels], stddev=W_std))
    b= tf.Variable(tf.random_normal([2 * spec.num_labels], stddev=b_std))
    logits= tf.matmul(lstmOut[:, -1, :], W) + b
    loss= tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=mixed_y)
    mixed_predict= tf.argmax(logits, 1)
    predict[perp_keys[0]]= mixed_predict % 2
    predict[perp_keys[1]]= mixed_predict // 2

    return tf.reduce_sum(loss)

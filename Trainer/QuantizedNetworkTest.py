# ------------------------------------------------------------------------------
#
# MIT License
#
# Copyright (c) 2018 Pooya Ronagh
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------

import sys, os, json
from time import time, strftime, localtime
import cPickle as pickle
from ModelExRecCNOT import *
from ModelSurface1EC import *

print ('Quantized network test for Steane_CNOT_D3 in LU-DND ...')
print ('Reading parameters of FF-2Hidden ...')
with open('../../Param/LookUp/Steane_CNOT_D3/2018-01-13-17-18-23.json') \
    as paramfile:
    param = json.load(paramfile)

if(param['env']['EC scheme']=='SurfaceD3'):
    import _SurfaceD3Lookup as lookup 
elif(param['env']['EC scheme']=='SurfaceD5'):
    import _SurfaceD5Lookup as lookup 
elif(param['env']['EC scheme']=='ColorD3'):
    import _ColorD3Lookup as lookup 
elif(param['env']['EC scheme']=='ColorD5'):
    import _ColorD5Lookup as lookup 
else:
    raise ValueError('Unknown circuit type.')
spec= lookup.Spec()

print ('Pickling 2e-04 pickle file as trainer ...')
start_time= time()
with open('../../Data/Pkl/LookUp/Steane_CNOT_D3/e-04/2.000e-04.pkl', 'rb') \
    as input_file:
    m = pickle.load(input_file)
print('Done in ' + '{0:.2f}'.format(time() - start_time) + 's.')

m.test_size= int(param['data']['test fraction'] * m.data_size)
m.train_size= m.data_size - m.test_size
m.num_batches= m.train_size // param['opt']['batch size']
m.spec= spec

num_hiddens= [spec.input_size] + param['nn']['num hidden'] + [spec.num_labels]
activations= []
for i in range(len(param['nn']['activations'])):
    if param['nn']['activations'][i]=='relu':
        activations.append(tf.nn.relu)
    elif param['nn']['activations'][i]=='sigmoid':
        activations.append(tf.nn.sigmoid)
    elif param['nn']['activations'][i]=='id':
        activations.append(tf.identity)
    elif param['nn']['activations'][i]=='tanh':
        activations.append(tf.tanh)
    else:
        raise Exception('Activation function not recognized.')

verbose= param['usr']['verbose']
batch_size= param['opt']['batch size']
learning_rate= param['opt']['learning rate']
num_iterations= param['opt']['iterations']
momentum_val= param['opt']['momentum']
decay_rate= param['opt']['decay']
pointer= randint(0, m.data_size - 1)
t_beg= (m.train_size + pointer) % m.data_size
num_test_batches= param['data']['num test batch'] if \
    'num test batch' in param['data'].keys() else 1
num_test_batches= 1
test_batch_size= m.test_size / num_test_batches

fault_rates= []
sig0= int(sys.argv[1])
sig1= int(sys.argv[2])

quantized_fault_rates= {}
for num_sig_dig in range(sig0, sig1 + 1):
    quantized_fault_rates[num_sig_dig]= []

for T in range(10):

    outfilename = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    prediction, test_beg= m.train(param, save= True, save_path=\
        param['env']['report folder'] + outfilename + '.ckpt')
    print('Testing ...'),
    start_time= time()
    result= m.num_logical_fault(prediction, test_beg)
    print('Done in ' + '{0:.2f}'.format(time() - start_time) + 's.')
    print('Result= ' + str(m.error_scale * result))
    fault_rates.append(m.error_scale * result)

    for num_sig_dig in range(sig0, sig1 + 1):

        tf.reset_default_graph()
        prediction= {}
        insig_dig= [2, num_sig_dig + 1, num_sig_dig + 1]
        with tf.Session() as session:
            saver = tf.train.import_meta_graph(\
            param['env']['report folder'] + outfilename + '.ckpt.meta')
            saver.restore(session, tf.train.latest_checkpoint(\
                param['env']['report folder']))

            x = {}
            for key in m.spec.err_keys:
                with tf.variable_scope(key):
                    x[key] = tf.placeholder(tf.float32, \
                        [None, m.spec.input_size], name= 'quant_x'+key)        

            abs_min= np.abs( \
                np.min([np.min(session.run(tf.trainable_variables()[i])) \
                    for i in range(len(tf.trainable_variables()))]))
            abs_max= np.abs( \
                np.max([np.max(session.run(tf.trainable_variables()[i])) \
                    for i in range(len(tf.trainable_variables()))]))
            print('Maximum of the weights in the network= ', str(+abs_min))
            print('Minimum of the weights in the network= ', str(-abs_max))

            dilation_factor= (2**(num_sig_dig-1)-1)/(np.max([abs_min, abs_max]))
            print('Dilation factor for quantization= ', str(dilation_factor))

            for k in range(len(tf.trainable_variables())):
                var= tf.trainable_variables()[k]
                new_var= tf.round(var * dilation_factor)
                assign_op= var.assign(new_var)
                session.run(assign_op)

            layer, logits, loss = {}, {}, {}
            tf_counter= 0
            quantized_predict= {}
            identities= {}
            max_val, min_val= {}, {}
            for key in spec.err_keys:
                layer[key]= []
                max_val[key]= []
                min_val[key]= []
                layer[key].append(x[key])
                with tf.variable_scope(key):
                    for l in range(len(num_hiddens)-1):
                        W= tf.trainable_variables()[tf_counter]
                        tf_counter+=1
                        b= tf.trainable_variables()[tf_counter]
                        tf_counter+=1
                        prod= tf.divide( \
                            tf.matmul(layer[key][-1], W) + b, 2**insig_dig[l])
                        prod= tf.sign(prod) * tf.floor(tf.abs(prod))
                        max_val[key].append(tf.reduce_max(prod))
                        min_val[key].append(tf.reduce_min(prod))
                        layer[key].append(activations[l](prod))
                    quantized_predict[key]= tf.argmax(layer[key][-1], 1, name='quant_predict')

            q_min_val= [0, 0, 0]
            q_max_val= [0, 0, 0]
            for j in tqdm.tqdm(range(num_test_batches)):
                beg= (j * test_batch_size + t_beg) % m.data_size
                if j==num_test_batches-1:
                    test_batch_size+= m.test_size % num_test_batches
                feed_dict={}
                for key in m.spec.err_keys:
                    feed_dict[x[key]]= \
                        cyc_pick(m.syn[key], beg, test_batch_size)
                for key in m.spec.err_keys:
                    for l in range(len(num_hiddens)-1):
                        new_min= session.run(min_val[key][l], feed_dict)
                        new_max= session.run(max_val[key][l], feed_dict)
                        if q_min_val[l] > new_min:
                            q_min_val[l]= new_min
                        if q_max_val[l] < new_max:
                            q_max_val[l]= new_max
            print('After quantization:')
            print('Maximum of the weights in the network= ', str(q_max_val))
            print('Minimum of the weights in the network= ', str(q_min_val))

            for j in tqdm.tqdm(range(num_test_batches)):
                beg= (j * test_batch_size + t_beg) % m.data_size
                if j==num_test_batches-1:
                    test_batch_size+= m.test_size % num_test_batches
                feed_dict={}
                for key in m.spec.err_keys:
                    feed_dict[x[key]]= \
                        cyc_pick(m.syn[key], beg, test_batch_size)
                for key in m.spec.err_keys:
                    prediction_result= session.run( \
                        quantized_predict[key], feed_dict)
                    if key in prediction.keys():
                        prediction[key]= np.append( \
                            prediction[key], prediction_result, axis=0)
                    else:
                        prediction[key]= prediction_result

            result= m.num_logical_fault(prediction, t_beg)
            print('Result= ' + str(m.error_scale * result))
            quantized_fault_rates[num_sig_dig].append(m.error_scale * result)
    
    print('Trial completed.')
    print(fault_rates)
    print(quantized_fault_rates)

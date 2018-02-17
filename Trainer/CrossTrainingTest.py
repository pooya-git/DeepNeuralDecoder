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

print ('Cross training test for Steane_CNOT_D5 in PE-DND ...')
print ('Reading parameters of FF-2Hidden ...')
with open('../../Param/PureError/Steane_CNOT_D5/2018-01-15-20-59-53.json') \
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

print ('Pickling 2e-03 pickle file as trainer ...')
start_time= time()
with open('../../Data/Pkl/PureError/Steane_CNOT_D5/e-04/2.000e-03.pkl', 'rb') \
	as input_file:
    m = pickle.load(input_file)
print('Done in ' + '{0:.2f}'.format(time() - start_time) + 's.')

m.test_size= int(param['data']['test fraction'] * m.data_size)
m.train_size= m.data_size - m.test_size
m.num_batches= m.train_size // param['opt']['batch size']
m.spec= spec

n= []
for filename in sorted(os.listdir(param['env']['pickle folder'])):
	print('Pickling the test set ' + str(filename))
	start_time= time()
	with open(param['env']['pickle folder'] + filename, 'rb') as input_file:
		n.append(pickle.load(input_file))
	print('Done in ' + '{0:.2f}'.format(time() - start_time) + 's.')

for i in range(len(n)):
    n[i].test_size= int(param['data']['test fraction'] * n[i].data_size)

vals= {}
for i in range(len(n)):
    vals[i]= []

verbose= param['usr']['verbose']
batch_size= param['opt']['batch size']
learning_rate= param['opt']['learning rate']
num_iterations= param['opt']['iterations']
momentum_val= param['opt']['momentum']
decay_rate= param['opt']['decay']

tf.reset_default_graph()
x, y, predict= {}, {}, {}
for key in m.spec.err_keys:
    with tf.variable_scope(key):
        x[key] = tf.placeholder(tf.float32, [None,m.spec.input_size])
        y[key] = tf.placeholder(tf.float32, [None,2])
keep_rate= tf.placeholder(tf.float32)
#cost= surface_conv3d_cost(param['nn'], m.spec, x, y, predict, keep_rate)
cost= m.cost_function(param['nn'], x, y, predict, keep_rate)
train = tf.train.RMSPropOptimizer(learning_rate, decay=decay_rate, \
    momentum=momentum_val).minimize(cost)
init = tf.global_variables_initializer()

for T in range(10):

    pointer= randint(0, m.data_size - 1)
    t_beg= (m.train_size + pointer) % m.data_size
    num_test_batches= param['data']['num test batch'] if \
        'num test batch' in param['data'].keys() else 1
    test_batch_size= m.test_size / num_test_batches

    predictions= []

    with tf.Session() as session:
        session.run(init)

        for i in tqdm.tqdm(range(num_iterations)):
            for j in range(m.num_batches):
                beg= (j * batch_size + pointer) % m.data_size
                feed_dict={}
                for key in m.spec.err_keys:
                    feed_dict[x[key]]= \
                        cyc_pick(m.syn[key], beg, batch_size)
                    feed_dict[y[key]]= \
                        cyc_pick(m.log_1hot[key], beg, batch_size)
                    feed_dict[keep_rate]= param['nn']['keep rate']
                session.run(train, feed_dict)

        for i in tqdm.tqdm(range(len(n))):

            prediction={}
            for j in range(num_test_batches):
                beg= (j * test_batch_size + t_beg) % m.data_size
                if j==num_test_batches-1:
                    test_batch_size+= m.test_size % num_test_batches
                feed_dict={}
                feed_dict[keep_rate]= 1.0
                for key in m.spec.err_keys:
                    feed_dict[x[key]]= \
                        cyc_pick(n[i].syn[key], beg, test_batch_size)
                for key in m.spec.err_keys:
                    prediction_result= session.run(predict[key], feed_dict)
                    if key in prediction.keys():
                        prediction[key]= np.append( \
                            prediction[key], prediction_result, axis=0)
                    else:
                        prediction[key]= prediction_result
            predictions.append(prediction)

    for i in tqdm.tqdm(range(len(n))):
        vals[i].append( \
        	n[i].error_scale * n[i].num_logical_fault(predictions[i], t_beg))
    
    print('Finished a new cross-testing procedure.')
    print(vals)

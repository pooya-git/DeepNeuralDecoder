# ------------------------------------------------------------------------------
# 
#    SurfaceD3 data preparation modules.
#
#    Copyright (C) 2017 Pooya Ronagh
# 
# ------------------------------------------------------------------------------

from builtins import range
import numpy as np
from util import y2indicator
import cPickle as pickle
import os


# --- d3 surface code generator matrices and look up table ---#

g= {}
L= {}
correctionMat= {}
g['Z'] = np.matrix([[1, 0, 0, 1, 0, 0, 0, 0, 0], \
                [0, 1, 1, 0, 1, 1, 0, 0, 0], \
                [0, 0, 0, 1, 1, 0, 1, 1, 0], \
                [0, 0, 0, 0, 0, 1, 0, 0, 1]]).astype(np.int32);
g['X'] = np.matrix([[1, 1, 0, 1, 1, 0, 0, 0, 0], \
                [0, 0, 0, 0, 0, 0, 1, 1, 0], \
                [0, 1, 1, 0, 0, 0, 0, 0, 0], \
                [0, 0, 0, 0, 1, 1, 0, 1, 1]]).astype(np.int32);
correctionMat['X'] = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0], \
                            [0, 0, 0, 0, 0, 0, 0, 0, 1], \
                            [0, 0, 0, 0, 0, 0, 1, 0, 0], \
                            [0, 0, 0, 0, 1, 1, 0, 0, 0], \
                            [0, 1, 0, 0, 0, 0, 0, 0, 0], \
                            [0, 0, 0, 0, 0, 1, 0, 0, 0], \
                            [0, 0, 0, 0, 1, 0, 0, 0, 0], \
                            [0, 0, 0, 0, 1, 0, 0, 0, 1], \
                            [1, 0, 0, 0, 0, 0, 0, 0, 0], \
                            [1, 0, 0, 0, 0, 0, 0, 0, 1], \
                            [0, 0, 0, 1, 0, 0, 0, 0, 0], \
                            [0, 0, 0, 1, 0, 0, 0, 0, 1], \
                            [1, 1, 0, 0, 0, 0, 0, 0, 0], \
                            [1, 0, 0, 0, 0, 1, 0, 0, 0], \
                            [1, 0, 0, 0, 1, 0, 0, 0, 0], \
                            [0, 0, 0, 1, 0, 1, 0, 0, 0]]).astype(np.int32);
correctionMat['Z'] = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0], \
                            [0, 0, 0, 0, 0, 1, 0, 0, 0], \
                            [0, 0, 1, 0, 0, 0, 0, 0, 0], \
                            [0, 1, 0, 0, 1, 0, 0, 0, 0], \
                            [0, 0, 0, 0, 0, 0, 1, 0, 0], \
                            [0, 0, 0, 0, 0, 0, 0, 1, 0], \
                            [0, 0, 1, 0, 0, 0, 1, 0, 0], \
                            [0, 0, 1, 0, 0, 0, 0, 1, 0], \
                            [1, 0, 0, 0, 0, 0, 0, 0, 0], \
                            [0, 0, 0, 0, 1, 0, 0, 0, 0], \
                            [0, 1, 0, 0, 0, 0, 0, 0, 0], \
                            [0, 1, 0, 0, 0, 1, 0, 0, 0], \
                            [1, 0, 0, 0, 0, 0, 1, 0, 0], \
                            [1, 0, 0, 0, 0, 0, 0, 1, 0], \
                            [0, 1, 0, 0, 0, 0, 1, 0, 0], \
                            [0, 1, 0, 0, 0, 0, 0, 1, 0]]).astype(np.int32);
L['Z'] = np.matrix([[1,0,0,0,1,0,0,0,1]]).astype(np.int32);
L['X'] = np.matrix([[0,0,1,0,1,0,1,0,0]]).astype(np.int32);

pauli_keys= ['X', 'Z']

def perp(key):
    if (key=='X'):
        return 'Z'
    if (key=='Z'):
        return 'X'
    print('Error: Unrecognized key!')

def correction_from_syn(syn, key):

    syn1= syn[0:4]
    syn2= syn[4:8]
    syn3= syn[8:12]
    if (max(syn1) + max(syn2) + max(syn3)<=1) :
        syndrome = np.zeros(4);
    elif np.array_equal(syn1, syn2) or np.array_equal(syn1, syn3):
        syndrome= syn1
    elif np.array_equal(syn2, syn3):
        syndrome= syn2
    else:
        syndrome= syn3
    correction_index= int(np.asscalar(np.dot([[8, 4, 2, 1]], syndrome)))
    return correctionMat[key][correction_index,:]

def syn_from_err(err, key):
    return np.dot(g[perp(key)], err.transpose()) % 2

def check_fault_from_err(err, key):

    syndrome= syn_from_err(err, key)
    correction_index= int(np.asscalar(np.dot([[8, 4, 2, 1]], syndrome)))
    correction= correctionMat[key][correction_index,:]
    errFinal = (correction + err) % 2
    logical_err = np.dot(L[perp(key)], errFinal.transpose()) % 2
    return logical_err


#--- Model creation modules ---#

def check_lu_fault(syn, err, key):
    
    correction= correction_from_syn(syn, key)
    errFinal = (correction + err) % 2
    return check_fault_from_err(errFinal, key)

class Data:

    def __init__(self, data, padding):
        self.syn= {}
        self.err= {}
        self.log= {}
        self.log_1hot= {}
        self.syn['X']= data['synX']
        self.syn['Z']= data['synZ']
        self.err['X']= data['errX']
        self.err['Z']= data['errZ']
        self.log['X']= []
        self.log['Z']= []
        for i in range(len(data['synX'])):
            self.log['X'].append(\
                check_lu_fault(data['synX'][i], data['errX'][i], 'X'))
        for i in range(len(data['synZ'])):
            self.log['Z'].append(\
                check_lu_fault(data['synZ'][i], data['errZ'][i], 'Z'))
        for key in pauli_keys:
            self.log[key]= np.array(self.log[key]).astype(np.float32)
            self.log_1hot[key]=y2indicator(self.log[key], 2).astype(np.float32)
        if (padding):
            self.input= {}
            self.input['X']= data['synX'].reshape(-1, 3, 2, 2, 1)
            self.input['X']= np.lib.pad(self.input['X'], \
                ((0,0), (0,0), (1,1), (1,1), (0, 0)), 'constant')
            self.input['Z']= data['synZ'].reshape(-1, 3, 2, 2, 1)
            self.input['Z']= np.lib.pad(self.input['Z'], \
                ((0,0), (0,0), (1,1), (1,1), (0, 0)), 'constant')

def io_data_factory(data, test_size, padding):

    train_data_arg = {key: data[key][:-test_size,] for key in data.keys()}
    test_data_arg  = {key: data[key][-test_size:,] for key in data.keys()}
    train_data = Data(train_data_arg, padding)
    test_data = Data(test_data_arg, padding)
    return train_data, test_data

def get_data(filename):

    data= {}
    for key in pauli_keys:
        data['syn'+key]= []
        data['err'+key]= []
    with open(filename) as file:
        first_line = file.readline();
        p, lu_avg, lu_std, data_size = first_line.split(' ')
        p= float(p)
        lu_avg= float(lu_avg)
        lu_std= float(lu_std)
        data_size= int(data_size)
        for line in file.readlines():
            line_list= line.strip('\n').split(' ')
            data['synX'].append([bit for bit in ''.join(line_list[0:3])])
            data['synZ'].append([bit for bit in ''.join(line_list[6:9])])
            data['errX'].append([bit for bit in line_list[5]])
            data['errZ'].append([bit for bit in line_list[11]])
    for key in data.keys():
        data[key]= np.array(data[key]).astype(np.float32)
    return data, p, lu_avg, lu_std, data_size

class Model:
    
    def __init__(self, data, test_fraction= 0.1, padding= False):
        data, p, lu_avg, lu_std, data_size = get_data(datafolder + filename)
        total_size= np.shape(data['synX'])[0]
        test_size= int(test_fraction * total_size)
        train_data, test_data = io_data_factory(data, test_size, padding)
        self.total_size = total_size
        self.p = p
        self.lu_avg = lu_avg
        self.lu_std = lu_std
        self.data_size = data_size
        self.test_size = test_size
        self.train_data= train_data
        self.test_data = test_data
        self.train_size= total_size - test_size
        self.error_scale= 1.0 * total_size/data_size


#--- check modules for the active model ---#

def error_rate(pred, truth):

    error_counter= 0.0
    for i in range(len(pred[pauli_keys[0]])):
        for key in pauli_keys:
            if (pred[key][i]!= truth[key][i]):
                error_counter+=1
                break
    return error_counter/len(pred[pauli_keys[0]])

def check_nn_fault(syn, err, key, pred):
    
    correction= correction_from_syn(syn, key)
    errFinal = (correction + err) % 2
    if (pred): correction+= L[key]
    return check_fault_from_err(errFinal, key)

def num_logical_fault(pred, truth):

    error_counter= 0.0
    for i in range(len(pred[pauli_keys[0]])):
        if not 1 in truth.syn['X'][i]: pred['X'][i]=0
        if not 1 in truth.syn['Z'][i]: pred['Z'][i]=0
        if (\
    check_nn_fault(truth.syn['X'][i], truth.err['X'][i], 'X', pred['X'][i]) or \
    check_nn_fault(truth.syn['Z'][i], truth.err['Z'][i], 'Z', pred['Z'][i])):
            error_counter+=1
    return error_counter/len(pred[pauli_keys[0]])
    
if __name__ == '__main__':

    datafolder= '../../Data/SurfaceD31EC/e-04/'
    file_list= os.listdir(datafolder)

    for filename in file_list:

        with open('../../Data/SurfaceD31EC_Pkl/e-04/'+ \
            filename.replace('.txt', '.pkl'), "wb") as output_file:
            print("Reading data from " + filename)
            model= Model(datafolder+ filename, padding= False)
            pickle.dump(model, output_file)

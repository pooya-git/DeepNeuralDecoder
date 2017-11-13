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

# The Surface code generator matrices and look up table
gZ = np.matrix([[1, 0, 0, 1, 0, 0, 0, 0, 0], \
                [0, 1, 1, 0, 1, 1, 0, 0, 0], \
                [0, 0, 0, 1, 1, 0, 1, 1, 0], \
                [0, 0, 0, 0, 0, 1, 0, 0, 1]]).astype(np.int32);
gX = np.matrix([[1, 1, 0, 1, 1, 0, 0, 0, 0], \
                [0, 0, 0, 0, 0, 0, 1, 1, 0], \
                [0, 1, 1, 0, 0, 0, 0, 0, 0], \
                [0, 0, 0, 0, 1, 1, 0, 1, 1]]).astype(np.int32);
XcorrectionMat = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0], \
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
ZcorrectionMat = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0], \
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
ZL = np.matrix([[1,0,0,0,1,0,0,0,1]]).astype(np.int32);
XL = np.matrix([[0,0,1,0,1,0,1,0,0]]).astype(np.int32);

err_keys= ['errX3', 'errZ3']
syn_keys= ['synX', 'synZ']

class Model:
    
    def __init__(self, data, test_fraction= 0.1):
        raw_data, p, lu_avg, lu_std, data_size = get_data(datafolder + filename)
        total_size= np.shape(raw_data['synX'])[0]
        test_size= int(test_fraction * total_size)
        train_data, test_data = io_data_factory(raw_data, test_size)
        self.total_size = total_size
        self.p = p
        self.lu_avg = lu_avg
        self.lu_std = lu_std
        self.data_size = data_size
        self.test_size = test_size
        self.train_data= train_data
        self.test_data = test_data
        self.train_size= total_size - test_size
        self.error_scale= 1.0*total_size/data_size

class Data:

    def __init__(self, data, padded= False):
        self.input= {}
        self.output= {}
        self.output_ind= {}
        self.err= {}
        if (padded):
            self.input['errX3']= data['synX'].reshape(-1, 3, 2, 2, 1)
            self.input['errX3']= np.lib.pad(self.input['errX3'], \
                ((0,0), (0,0), (1,1), (1,1), (0, 0)), 'constant')
            self.input['errZ3']= data['synZ'].reshape(-1, 3, 2, 2, 1)
            self.input['errZ3']= np.lib.pad(self.input['errZ3'], \
                ((0,0), (0,0), (1,1), (1,1), (0, 0)), 'constant')
        else:
            self.input['errX3']= data['synX']
            self.input['errZ3']= data['synZ']
        self.err['errX3']= data['errX3']
        self.err['errZ3']= data['errZ3']
        self.output['errX3']= []
        self.output['errZ3']= []
        for i in range(len(data['synX'])):
            self.output['errX3'].append(\
                logical(data['synX'][i], data['errX3'][i], 'errX3'))
        for i in range(len(data['synZ'])):
            self.output['errZ3'].append(\
                logical(data['synZ'][i], data['errZ3'][i], 'errZ3'))
        for key in err_keys:
            self.output[key]= np.array(self.output[key]).astype(np.float32)
            self.output_ind[key]=y2indicator(\
                self.output[key], 2).astype(np.float32)

def io_data_factory(data, test_size):

    train_data_arg = {key: data[key][:-test_size,] for key in data.keys()}
    test_data_arg  = {key: data[key][-test_size:,] for key in data.keys()}
    train_data = Data(train_data_arg, padded= True)
    test_data = Data(test_data_arg, padded= True)
    return train_data, test_data

def error_rate(prediction, truth):

    error_counter= 0.0
    for i in range(len(prediction[err_keys[0]])):
        for key in err_keys:
            if (prediction[key][i]!= truth[key][i]):
                error_counter+=1
                break
    return error_counter/len(prediction[err_keys[0]])

def num_logical_fault(prediction, test_data):

    error_counter= 0.0
    for i in range(len(prediction[err_keys[0]])):
        if (check_logical_fault(test_data.input['errX3'][i], \
            test_data.err['errX3'][i], 'errX3', prediction['errX3'][i])
            or 
            check_logical_fault(test_data.input['errZ3'][i], \
            test_data.err['errZ3'][i], 'errZ3', prediction['errZ3'][i])):
            error_counter+=1
    return error_counter/len(prediction[err_keys[0]])
    
def check_logical_fault(syn, err, key, pred):
    
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
    
    if (key=='errX3'):
        correction = XcorrectionMat[correction_index,:]
        if (pred): correction+= XL
        errFinal = (correction + err) % 2
        logical_err = np.dot(ZL, errFinal.transpose()) % 2
        return logical_err
    elif (key=='errZ3'):
        correction = ZcorrectionMat[correction_index,:]
        if (pred): correction+= ZL
        errFinal = (correction + err) % 2
        logical_err = np.dot(XL, errFinal.transpose()) % 2
        return logical_err

    else: print('Key not recognized.')

def logical(syn, err, key):
    
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
    
    if (key=='errX3'):
        correction = XcorrectionMat[correction_index,:]
        errFinal = (correction + err) % 2
        logical_err = np.dot(ZL, errFinal.transpose()) % 2
        return logical_err
    elif (key=='errZ3'):
        correction = ZcorrectionMat[correction_index,:]
        errFinal = (correction + err) % 2
        logical_err = np.dot(XL, errFinal.transpose()) % 2
        return logical_err

    else: print('Key not recognized.')

def get_data(filename):

    data= {}
    for key in syn_keys:
        data[key]= []
    for key in err_keys:
        data[key]= []
    with open(filename) as file:
        first_line = file.readline();
        p, lu_avg, lu_std, data_size = first_line.split(' ')
        p= float(p)
        lu_avg= float(lu_avg)
        lu_std= float(lu_std)
        data_size= int(data_size)
        for line in file.readlines():
            line_list= line.split(' ')
            data['synX'].append([bit for bit in ''.join(line_list[0:3])])
            data['synZ'].append([bit for bit in ''.join(line_list[6:9])])
            data['errX3'].append([int(line_list[5],2)])
            data['errZ3'].append([int(line_list[11],2)])
    for key in data.keys():
        data[key]= np.array(data[key]).astype(np.float32)
    return data, p, lu_avg, lu_std, data_size


if __name__ == '__main__':

    datafolder= '../Data/SurfaceD3/e-04/'
    file_list= os.listdir(datafolder)

    for filename in file_list:

        with open('../Data/SurfaceD3ConvPkl/e-04/'+ \
            filename.replace('.txt', '.pkl'), "wb") as output_file:
            print("Reading data from " + filename)
            model= Model(datafolder+ filename)
            pickle.dump(model, output_file)

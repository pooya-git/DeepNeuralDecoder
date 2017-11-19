# ------------------------------------------------------------------------------
# 
#    CNOTExRec data analysis modules.
#
#    Copyright (C) 2017 Pooya Ronagh
# 
# ------------------------------------------------------------------------------

from builtins import range
import numpy as np
import sys
from util import y2indicator
import sys
import os
import cPickle as pickle

#-- The 7 qubit CSS code generator --#

G= np.matrix([[0,0,0,1,1,1,1], \
              [0,1,1,0,0,1,1], \
              [1,0,1,0,1,0,1]]).astype(np.int32)

err_keys= ['errX3', 'errX4', 'errZ3', 'errZ4']
syn_keys= ['synX12', 'synX34', 'synZ12', 'synZ34']

class Model:
    
    def __init__(self, path):
        raw_data, p, lu_avg, lu_std, size = get_data(path)
        self.p = p
        self.lu_avg = lu_avg
        self.lu_std = lu_std
        self.total_size = size
        self.data = Data(raw_data)
        self.data_size = np.shape(raw_data['synX12'])[0]

class Data:

    def __init__(self, data):
        
        self.syn= {}
        synX= np.concatenate(\
            (data['synX12'], data['synX34']), axis= 1).reshape(-1, 2, 6)
        synZ= np.concatenate(\
            (data['synZ12'], data['synZ34']), axis= 1).reshape(-1, 2, 6)
        self.syn['errX3']= synX
        self.syn['errX4']= synX
        self.syn['errZ3']= synZ
        self.syn['errZ4']= synZ

        rep_X1= lookup_correction(data['synX12'][:,0:3])
        rep_X2= lookup_correction(data['synX12'][:,3:6])
        rep_Z1= lookup_correction(data['synZ12'][:,0:3])
        rep_Z2= lookup_correction(data['synZ12'][:,3:6])
        rep_X3= lookup_correction(data['synX34'][:,0:3])
        rep_X4= lookup_correction(data['synX34'][:,3:6])
        rep_Z3= lookup_correction(data['synZ34'][:,0:3])
        rep_Z4= lookup_correction(data['synZ34'][:,3:6])

        self.rec= {}
        self.rec['errX3']= ((data['errX3'] + rep_X1 + \
            lookup_correction_from_err(rep_X1 + rep_X3)) % 2).astype(np.int8)
        self.rec['errZ3']= ((data['errZ3'] + rep_Z1 + rep_Z2 + \
            lookup_correction_from_err(\
                rep_Z1 + rep_Z2 + rep_Z3)) % 2).astype(np.int8)
        self.rec['errX4']= ((data['errX4'] + rep_X1 + rep_X2 + \
            lookup_correction_from_err(\
                rep_X1 + rep_X2 + rep_X4)) % 2).astype(np.int8)    
        self.rec['errZ4']= ((data['errZ4'] + rep_Z3 + \
            lookup_correction_from_err(rep_Z3 + rep_Z4)) % 2).astype(np.int8)

        self.err_1hot= {}
        for key in err_keys:
            err = np.array(\
            np.sum((self.rec[key] \
                + lookup_correction_from_err(self.rec[key])) % 2, axis= 1)\
                % 2).transpose()
            self.err_1hot[key]=y2indicator(err, 2).astype(np.int8)
        
def syndrome(err):
    return np.dot(err, G.transpose()) % 2

def lookup_correction(syn):
    correction_index= np.dot(syn, [[4], [2], [1]]) - 1
    return y2indicator(correction_index, 7)

def lookup_correction_from_err(err):
    syn= syndrome(err)
    return lookup_correction(syn)

def find_logical_fault(err):

    syndrome= np.dot(G, err.transpose()) % 2
    correction_index= np.dot([[4, 2, 1]], syndrome.transpose()) - 1
    correction = y2indicator(correction_index, 7)
    coset= (err + correction) % 2
    logical_err= np.sum(coset) % 2
    return logical_err

def num_logical_fault(pred, syn, rec):

    error_counter= 0.0
    for i in range(len(pred[err_keys[0]])):
        for key in err_keys:
            if not 1 in syn[key][i]: pred[key][i]=0
            if (find_logical_fault(pred[key][i]*np.ones(7) + rec[key][i] % 2)):
                error_counter+=1
                break
    return error_counter/len(pred[err_keys[0]])

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
            line_list= line.strip('\n').split(' ')
            data['synX12'].append([bit for bit in ''.join(line_list[0:2])])
            data['synX34'].append([bit for bit in ''.join(line_list[2:4])])
            data['synZ12'].append([bit for bit in ''.join(line_list[8:10])])
            data['synZ34'].append([bit for bit in ''.join(line_list[10:12])])
            data['errX3'].append([bit for bit in line_list[6]])
            data['errX4'].append([bit for bit in line_list[7]])
            data['errZ3'].append([bit for bit in line_list[14]])
            data['errZ4'].append([bit for bit in line_list[15]])
    for key in data.keys():
        data[key]= np.array(data[key]).astype(np.int8)
    return data, p, lu_avg, lu_std, data_size

if __name__ == '__main__':

    datafolder= '../../Data/SteaneCNOT/e-04/'
    file_list= os.listdir(datafolder)

    for filename in file_list:

        with open('../../Data/SteaneCNOT_Pkl/e-04/'+ \
            filename.replace('.txt', '.pkl'), "wb") as output_file:
            print("Reading data from " + filename)
            model= Model(datafolder+ filename)
            pickle.dump(model, output_file)
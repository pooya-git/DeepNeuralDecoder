# ------------------------------------------------------------------------------
# 
#    Surface1EC fault tolerant error correction model.
#
#    Copyright (C) 2017 Pooya Ronagh
# 
# ------------------------------------------------------------------------------

from Model import *
import numpy as np
import util

class Surface1EC(Model):

    def __init__(self, path, spec):

        super(Surface1EC, self).__init__(path, spec)

    def get_data(self, path):

        data= {}
        for key in self.spec.err_keys:
            data['syn'+key]= []
            data['err'+key]= []
        with open(path) as file:
            first_line = file.readline();
            p, lu_avg, lu_std, data_size = first_line.split(' ')
            p= float(p)
            lu_avg= float(lu_avg)
            lu_std= float(lu_std)
            data_size= int(data_size)
            for line in file.readlines():
                line_list= line.strip('\n').split(' ')
                data['synX'].append([bit for bit in \
                    ''.join(line_list[0:self.spec.d])])
                data['synZ'].append([bit for bit in \
                    ''.join(line_list[2*self.spec.d:3*self.spec.d])])
                data['errX'].append([bit for bit in line_list[2*self.spec.d-1]])
                data['errZ'].append([bit for bit in line_list[4*self.spec.d-1]])
        for key in data.keys():
            data[key]= np.matrix(data[key]).astype(np.int8)
        return data, p, lu_avg, lu_std, data_size

    def init_data(self, raw_data):

        self.syn= {}
        self.err= {}
        self.log_1hot= {}
        self.syn['X']= raw_data['synX']
        self.syn['Z']= raw_data['synZ']
        self.err['X']= raw_data['errX']
        self.err['Z']= raw_data['errZ']
        log= {}
        log['X']= np.matrix(np.zeros([self.data_size, 1])).astype(np.int8)
        log['Z']= np.matrix(np.zeros([self.data_size, 1])).astype(np.int8)
        counter= 0
        for i in range(self.data_size):
            log['X'][i]= self.check_lu_fault(\
                    raw_data['synX'][i,:], raw_data['errX'][i,:], 'X')
            log['Z'][i]= self.check_lu_fault(\
                    raw_data['synZ'][i,:], raw_data['errZ'][i,:], 'Z')
            if (log['X'][i] + log['Z'][i] >= 1):
                counter+=1
        print counter
        for key in self.spec.err_keys:
            self.log_1hot[key]=util.y2indicator(log[key], 2).astype(np.int8)

    def correction_from_syn(self, syn, key):

        syn_dic= {}
        for i in range(self.spec.d):
            syn_dic[i]= syn[0, self.spec.syn_size*i : self.spec.syn_size*(i+1)]
        if (self.spec.d==3):
            syndrome_index= 0
            syn_eq_flag= False
            for i in range(self.spec.d-1):
                if np.array_equal(syn_dic[i], syn_dic[i+1]):
                    syn_eq_flag= True
                    syndrome_index= i
            if not syn_eq_flag:
                syndrome_index= self.spec.d-1
            syndrome= syn_dic[syndrome_index]
        elif (self.spec.d==5):
            raise Exception('Method not implemented.')

        correction_index= np.asscalar(util.vec_to_index(syndrome))
        return self.spec.correctionMat[key][correction_index]

    def syn_from_err(self, err, key):
        return np.dot(err, self.spec.g[util.perp(key)].transpose()) % 2

    def check_fault_from_err(self, err, key):

        syndrome= self.syn_from_err(err, key)
        correction_index= np.asscalar(util.vec_to_index(syndrome))
        correction= self.spec.correctionMat[key][correction_index]
        errFinal= (correction + err) % 2
        logical_err= np.dot(self.spec.L[util.perp(key)], errFinal.transpose())%2
        return np.asscalar(logical_err)

    def check_lu_fault(self, syn, err, key):
        
        correction= self.correction_from_syn(syn, key)
        errFinal = (correction + err) % 2
        return self.check_fault_from_err(errFinal, key)

    def check_nn_fault(self, syn, err, key, pred_val):
        
        correction= self.correction_from_syn(syn, key)
        errFinal = (correction + err + pred_val * self.spec.L[key]) % 2
        return self.check_fault_from_err(errFinal, key)

    def num_logical_fault(self, pred, t_beg):

        error_counter= 0.0
        for i in range(self.test_size):
            t_index= (i + t_beg) % self.data_size
            if not 1 in self.syn['X'][i]: pred['X'][i]=0
            if not 1 in self.syn['Z'][i]: pred['Z'][i]=0
            if (self.check_nn_fault(self.syn['X'][t_index], \
                    self.err['X'][t_index], 'X', np.asscalar(pred['X'][i])) or \
                self.check_nn_fault(self.syn['Z'][t_index], \
                    self.err['Z'][t_index], 'Z', np.asscalar(pred['Z'][i]))):
                error_counter+=1
        return error_counter/self.test_size
        
# ------------------------------------------------------------------------------
# 
#    CNOTExRec fault tolerant error correction model.
#
#    Copyright (C) 2017 Pooya Ronagh
# 
# ------------------------------------------------------------------------------

from Model import *
import numpy as np
import util

class ExRecCNOT(Model):

    def __init__(self, path, spec):

        super(ExRecCNOT, self).__init__(path, spec)

    def get_data(self, path):

        data= {}
        for key in self.spec.syn_keys:
            data[key]= []
        for key in self.spec.err_keys:
            data[key]= []
        with open(path) as file:
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
            data[key]= np.matrix(data[key]).astype(np.int8)
        return data, p, lu_avg, lu_std, data_size

    def init_data(self, raw_data):
        
        self.syn= {}
        synX= np.concatenate((raw_data['synX12'], raw_data['synX34']), axis= 1)
        synZ= np.concatenate((raw_data['synZ12'], raw_data['synZ34']), axis= 1)
        self.syn['errX3']= synX
        self.syn['errX4']= synX
        self.syn['errZ3']= synZ
        self.syn['errZ4']= synZ

        rep_X1=self.lookup_correction(raw_data['synX12'][:,:self.spec.syn_size])
        rep_X2=self.lookup_correction(raw_data['synX12'][:,self.spec.syn_size:])
        rep_Z1=self.lookup_correction(raw_data['synZ12'][:,:self.spec.syn_size])
        rep_Z2=self.lookup_correction(raw_data['synZ12'][:,self.spec.syn_size:])
        rep_X3=self.lookup_correction(raw_data['synX34'][:,:self.spec.syn_size])
        rep_X4=self.lookup_correction(raw_data['synX34'][:,self.spec.syn_size:])
        rep_Z3=self.lookup_correction(raw_data['synZ34'][:,:self.spec.syn_size])
        rep_Z4=self.lookup_correction(raw_data['synZ34'][:,self.spec.syn_size:])

        self.rec= {}
        self.rec['errX3']= np.matrix((raw_data['errX3'] + rep_X1 + \
            self.lookup_correction_from_err((rep_X1 + rep_X3) % 2)) \
            % 2).astype(np.int8)
        self.rec['errZ3']= np.matrix((raw_data['errZ3'] + rep_Z1 + rep_Z2 + \
            self.lookup_correction_from_err((rep_Z1 + rep_Z2 + rep_Z3) % 2)) \
            % 2).astype(np.int8)
        self.rec['errX4']= np.matrix((raw_data['errX4'] + rep_X1 + rep_X2 + \
            self.lookup_correction_from_err((rep_X1 + rep_X2 + rep_X4) % 2)) \
            % 2).astype(np.int8)    
        self.rec['errZ4']= np.matrix((raw_data['errZ4'] + rep_Z2 + \
            self.lookup_correction_from_err((rep_Z2 + rep_Z4) % 2)) \
            % 2).astype(np.int8)

        self.log_1hot= {}
        for key in self.spec.err_keys:
            err = np.matrix(\
            np.sum(self.rec[key] + \
                self.lookup_correction_from_err(self.rec[key], axis= 1)) % 2)
            self.log_1hot[key]= util.y2indicator(err, 2).astype(np.int8)

    def syn_from_err(self, err):

        return np.dot(err, self.spec.G.transpose()) % 2

    def lookup_correction(self, syn):

        correction_index= util.vec_to_index(syn)
        correction= self.spec.correctionMat[\
                    correction_index.transpose().tolist()]
        assert (np.shape(syn)[1] == self.spec.syn_size)
        assert (np.shape(correction_index)[1] == 1)
        assert (np.shape(correction_index)[0] == np.shape(syn)[0])
        return correction

    def lookup_correction_from_err(self, err):

        syn= self.syn_from_err(err)
        return self.lookup_correction(syn)

    def check_log_fault(self, err):

        correction = self.lookup_correction_from_err(err)
        coset= (err + correction) % 2
        logical_err= np.sum(coset) % 2
        return logical_err

    def num_logical_fault(self, pred, t_beg):

        error_counter= 0.0
        for i in range(self.test_size):
            t_index= (i + t_beg) % self.data_size
            for key in self.spec.err_keys:
                if not 1 in self.syn[key][t_index]: pred[key][i]=0
                if (self.check_log_fault((\
                pred[key][i] * np.ones((1,self.spec.num_qubit)).astype(np.int8)\
                + self.rec[key][t_index]) % 2)):
                    error_counter+=1
                    break
        return error_counter/len(pred[self.spec.err_keys[0]])


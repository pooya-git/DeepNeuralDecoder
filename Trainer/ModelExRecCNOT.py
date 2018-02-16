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
                data['synZ12'].append([bit for bit in ''.join(line_list[6:8])])
                data['synZ34'].append([bit for bit in ''.join(line_list[8:10])])
                data['errX3'].append([bit for bit in line_list[4]])
                data['errX4'].append([bit for bit in line_list[5]])
                data['errZ3'].append([bit for bit in line_list[10]])
                data['errZ4'].append([bit for bit in line_list[11]])
        for key in data.keys():
            data[key]= np.matrix(data[key]).astype(np.int8)
        return data, p, lu_avg, lu_std, data_size

    def init_syn(self, raw_data):
        
        self.syn= {}
        synX= np.concatenate((raw_data['synX12'], raw_data['synX34']), axis= 1)
        synZ= np.concatenate((raw_data['synZ12'], raw_data['synZ34']), axis= 1)
        self.syn['errX3']= synX
        self.syn['errX4']= synX
        self.syn['errZ3']= synZ
        self.syn['errZ4']= synZ

    def abstract_init_rec(self, raw_data, abs_corr, err_corr):

        rep_X1= abs_corr(raw_data['synX12'][:,:self.spec.syn_size],'errX3')
        rep_X2= abs_corr(raw_data['synX12'][:,self.spec.syn_size:],'errX4')
        rep_Z1= abs_corr(raw_data['synZ12'][:,:self.spec.syn_size],'errZ3')
        rep_Z2= abs_corr(raw_data['synZ12'][:,self.spec.syn_size:],'errZ4')
        rep_X3= abs_corr(raw_data['synX34'][:,:self.spec.syn_size],'errX3')
        rep_X4= abs_corr(raw_data['synX34'][:,self.spec.syn_size:],'errX4')
        rep_Z3= abs_corr(raw_data['synZ34'][:,:self.spec.syn_size],'errZ3')
        rep_Z4= abs_corr(raw_data['synZ34'][:,self.spec.syn_size:],'errZ4')

        self.rec= {}
        self.rec['errX3']= np.matrix((raw_data['errX3'] + rep_X1 + \
            err_corr((rep_X1+rep_X3) % 2, 'errX3')) % 2).astype(np.int8)
        self.rec['errZ3']= np.matrix((raw_data['errZ3'] + rep_Z1 + rep_Z2 + \
            err_corr((rep_Z1+rep_Z2+rep_Z3) % 2, 'errZ3')) % 2).astype(np.int8)
        self.rec['errX4']= np.matrix((raw_data['errX4'] + rep_X1 + rep_X2 + \
            err_corr((rep_X1+rep_X2+rep_X4) % 2, 'errX4')) % 2).astype(np.int8)    
        self.rec['errZ4']= np.matrix((raw_data['errZ4'] + rep_Z2 + \
            err_corr((rep_Z2+rep_Z4) % 2, 'errZ4')) % 2).astype(np.int8)

class LookUpExRecCNOT(ExRecCNOT):

    def __init__(self, path, spec):

        super(LookUpExRecCNOT, self).__init__(path, spec)

    def init_rec(self, raw_data):

        self.abstract_init_rec(\
            raw_data, self.lookup_correction, self.lookup_correction_from_error)

class PureErrorExRecCNOT(ExRecCNOT):

    def __init__(self, path, spec):

        super(PureErrorExRecCNOT, self).__init__(path, spec)

    def init_rec(self, raw_data):

        self.abstract_init_rec(\
            raw_data, self.pure_correction, self.pure_correction_from_error)

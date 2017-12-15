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

    def init_syn(self, raw_data):

        self.syn= {}
        self.syn['X']= raw_data['synX']
        self.syn['Z']= raw_data['synZ']

    def choose_syndrome(self, syn):

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
            return syndrome
        elif (self.spec.d>=5):
            raise Exception('Method not implemented.')

    def abstract_init_rec(self, raw_data, abs_corr):

        self.rec= {}
        for key in self.spec.err_keys:
            rep_syn= np.matrix(\
                np.zeros([self.data_size, self.spec.syn_size])).astype(np.int8)
            for i in range(self.data_size):
                rep_syn[i]= self.choose_syndrome(self.syn[key][i])
            self.rec[key]= raw_data['err' + key] + abs_corr(rep_syn, key)

class LookUpSurface1EC(Surface1EC):

    def __init__(self, path, spec):

        super(LookUpSurface1EC, self).__init__(path, spec)

    def init_rec(self, raw_data):

        self.abstract_init_rec(raw_data, self.lookup_correction)

class PureErrorSurface1EC(Surface1EC):

    def __init__(self, path, spec):

        super(PureErrorSurface1EC, self).__init__(path, spec)

    def init_rec(self, raw_data):

        self.abstract_init_rec(raw_data, self.pure_correction)

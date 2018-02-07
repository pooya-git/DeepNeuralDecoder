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

        ### If in the simulation the errors are not padded ###
        # if (self.spec.d >= 5):
        #     for key in self.spec.err_keys:
        #         data['fullErr'+key]= []
        
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
                    ''.join(line_list[0:self.spec.num_syn])])
                data['synZ'].append([bit for bit in \
                    ''.join(line_list[2*self.spec.num_syn \
                                     :3*self.spec.num_syn])])
                data['errX'].append([bit for bit in \
                    line_list[2*self.spec.num_syn - 1]])
                data['errZ'].append([bit for bit in \
                    line_list[4*self.spec.num_syn - 1]])

                ### If in the simulation the errors are not padded ###
                # if (self.spec.d >= 5):
                #     data['fullErrX'].append([bit for bit in \
                #         ''.join(line_list[1*self.spec.num_syn \
                #                          :2*self.spec.num_syn])])
                #     data['fullErrZ'].append([bit for bit in \
                #         ''.join(line_list[3*self.spec.num_syn \
                #                          :4*self.spec.num_syn])])
                    
        for key in data.keys():
            data[key]= np.matrix(data[key]).astype(np.int8)
        return data, p, lu_avg, lu_std, data_size

    def init_syn(self, raw_data):

        self.syn= {}
        self.syn['X']= raw_data['synX']
        self.syn['Z']= raw_data['synZ']

    def choose_syndrome(self, syn):

        syn_dic= {}
        for i in range(self.spec.num_syn):
            syn_dic[i]= syn[0, self.spec.syn_size*i : self.spec.syn_size*(i+1)]

        if (self.spec.d==3):
            assert(self.spec.d == self.spec.num_syn)
            syndrome_index= 0
            syn_eq_flag= False
            for i in range(self.spec.num_syn-1):
                if np.array_equal(syn_dic[i], syn_dic[i+1]):
                    syn_eq_flag= True
                    syndrome_index= i
            if not syn_eq_flag:
                syndrome_index= self.spec.num_syn-1

        elif (self.spec.d>=5):
            n_diff= 0
            syn_eq_num= 1
            syndrome_index= 0
            last_round= False
            syn_eq_flag= False
            n_diff_prev_update= False
            t= np.floor((self.spec.d-1)/2)
            for i in range(1, self.spec.num_syn):
                syn_eq_flag= np.array_equal(syn_dic[i-1], syn_dic[i])
                if not syn_eq_flag:
                    if not n_diff_prev_update:
                        n_diff+= 1
                        n_diff_prev_update= True
                        syn_eq_num= 0
                    else:
                        n_diff_prev_update= False
                        syn_eq_num= 1
                else:
                    n_diff_prev_update= False
                    syn_eq_num+= 1
                if (n_diff == t):
                    syndrome_index= i+1
                    break
                if (syn_eq_num == t - n_diff + 1):
                    syndrome_index= i
                    break
                
        return syn_dic[syndrome_index], syndrome_index

    def abstract_init_rec(self, raw_data, abs_corr):

        self.rec= {}
        rep_syn= {}

        ### Do this, if in the simulation the errors are not already padded. ###
        # max_syndex= np.matrix(np.zeros([self.data_size, 1])).astype(np.int8)

        for key in self.spec.err_keys:
            rep_syn[key]= np.matrix(\
                np.zeros([self.data_size, self.spec.syn_size])).astype(np.int8)
            
            for i in range(self.data_size):
                rep_syn[key][i], syndex= self.choose_syndrome(self.syn[key][i])

                if (self.spec.d >= 5):

                    ## Update max_syndex if it is used. ##
                    # if (syndex > max_syndex[i]):
                    #     max_syndex[i]= syndex

                    t= np.floor((self.spec.d-1)/2)
                    for j in range(syndex, int((t+1)*(t+2)/2)):
                        self.syn[key][j, \
                            self.spec.syn_size*j : self.spec.syn_size*(j+1)]= \
                            rep_syn[key][i]
                    
        ### Do this, if in the simulation the errors are not already padded. ###
        # if (self.spec.d >= 5):
        #     for key in self.spec.err_keys:
        #         for i in range(self.data_size):
        #             err_index= np.asscalar(max_syndex[i])
        #             raw_data['err' + key][i]= raw_data['fullErr' + key][ \
        #                 i, err_index * self.spec.d * self.spec.d : \
        #                    (err_index + 1) * self.spec.d * self.spec.d]

        for key in self.spec.err_keys:            
            self.rec[key]= raw_data['err' + key] + abs_corr(rep_syn[key], key)

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

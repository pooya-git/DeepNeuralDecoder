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

import numpy as np

class Spec:

    def __init__(self):

        self.num_qubit= 7
        self.syn_size= 3
        self.num_syn= 4
        self.input_size= 12
        self.num_labels= 2

        self.num_epochs= 2
        self.lstm_input_size= 6

        self.err_keys= ['errX3','errX4', 'errZ3', 'errZ4']
        self.syn_keys= ['synX12', 'synX34', 'synZ12', 'synZ34']
        self.perp_keys= [('errX3', 'errZ3'), ('errX4', 'errZ4')]

        #-- The 7 qubit CSS code generator --#

        self.L= {}
        for key in self.err_keys:
            self.L[key]= np.matrix([\
            [1,1,1,1,1,1,1]]).astype(np.int8)

        self.G= {}
        for key in self.err_keys:
            self.G[key] = np.matrix([ \
                [0,0,0,1,1,1,1], \
                [0,1,1,0,0,1,1], \
                [1,0,1,0,1,0,1]]).astype(np.int8)

        self.T= {}
        for key in ['errX3','errX4']:
            self.T[key]= np.matrix([\
                [0,0,0,1,0,0,0], \
                [0,1,0,0,0,0,0], \
                [1,0,0,0,0,0,0]]).astype(np.int8)

        for key in ['errZ3', 'errZ4']:
            self.T[key]= np.matrix([\
                [0,0,1,0,0,0,1], \
                [0,0,0,0,1,0,1], \
                [0,0,0,0,0,1,1]]).astype(np.int8)

        self.correctionMat= {}
        for key in self.err_keys:
            self.correctionMat[key] = np.matrix([ \
                [0,0,0,0,0,0,0], \
                [1,0,0,0,0,0,0], \
                [0,1,0,0,0,0,0], \
                [0,0,1,0,0,0,0], \
                [0,0,0,1,0,0,0], \
                [0,0,0,0,1,0,0], \
                [0,0,0,0,0,1,0], \
                [0,0,0,0,0,0,1]]).astype(np.int8)
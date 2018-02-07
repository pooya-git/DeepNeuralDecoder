#
# Author: Pooya Ronagh (2017)
# All rights reserved.
#
# This script takes input data from Christopher Chamberland's Matlab code
# and strips X syndrome and error data. If there is any nonzero bit in X data
# it prints a lines of the form 
# (SynX1, SynX2, SynX3, SynX4) + ' ' + (ErrX3, ErrX4)
#

from __future__ import print_function, division
from builtins import range
import numpy as np
import sys
import os
import json

headers= [[1e-4, 5.38e-5, 6.1415e-6, 92372189], \
          [2e-4, 2.308e-4, 1.2766e-5, 46939591], \
          [3e-4, 4.861e-4, 1.8796e-5, 31744132], \
          [4e-4, 8.920e-4, 2.575e-5, 23756420], \
          [5e-4, 0.0015, 3.8869e-5, 19119486], \
          [6e-4, 0.0020, 3.8869e-5, 16066622], \
          [7e-4, 0.0027, 4.5300e-5, 14153973]]

def run(syn_folder, err_folder, output_folder, filename, header_line):

    print_epoch= 100000

    with open(syn_folder + filename) as file:
        print(filename + ' ...')
        syn_lines = file.readlines()

    with open(err_folder + filename) as file:
        print(filename + ' ...')
        err_lines = file.readlines()

    assert(len(syn_lines)==2*len(err_lines))
    print(len(syn_lines)/4)
    outstream= open(output_folder + filename, 'w+')
    outstream.write(' '.join([str(elt) for elt in header_line]) + '\n')
    for line_num in range(len(syn_lines)/4):
        if (not line_num % print_epoch):
            print(line_num)
        synx= []
        errx= []
        synz= []
        errz= []
        for i in range(4):
            xz_syn_str=  ''.join(syn_lines[4*line_num + i].split('\t')).strip()
            synx.append(xz_syn_str[0:3])
            synz.append(xz_syn_str[3:6])
        for i in range(2):
            xz_err_str=  ''.join(err_lines[2*line_num + i].split('\t')).strip()
            errx.append(xz_err_str[0:7])
            errz.append(xz_err_str[7:14])
        result= ' '.join(synx) + ' ' + ' '.join(errx) \
        + ' ' + ' '.join(synz) + ' ' + ' '.join(errz)
        if '1' in result:
            outstream.write(result + '\n')
        else:
            print('Warning at line ' + str(line_num))
            print(result)

    outstream.close()

if __name__ == '__main__':

    counter= 0
    for filename in sorted(os.listdir(sys.argv[1])):
        run(sys.argv[1], sys.argv[2], sys.argv[3], filename, headers[counter])
        sys.stdout.flush()
        counter+=1

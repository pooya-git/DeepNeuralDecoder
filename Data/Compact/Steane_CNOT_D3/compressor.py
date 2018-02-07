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

headers= [[1e-4, 4.69e-5, 5.9478e-6, 104742878], \
          [2e-4, 1.949e-4, 1.2393e-4, 53247045], \
          [3e-4, 4.3090e-4, 1.8388e-5, 35913636], \
          [4e-4, 7.7080e-4, 2.4592e-5, 26833948], \
          [5e-4, 0.0012, 3.0853e-5, 21528378], \
          [6e-4, 0.0017, 3.6771e-5, 18058060], \
          [7e-4, 0.0023, 4.2612e-5, 15939483]]

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

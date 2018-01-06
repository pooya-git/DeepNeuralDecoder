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

headers= [[1e-4, 4.69e-5, 5.9478e-6, 116073323], \
          [2e-4, 1.949e-4, 1.2393e-4, 65002532], \
          [3e-4, 4.3090e-4, 1.8388e-5, 45123855], \
          [4e-4, 7.7080e-4, 2.4592e-5, 32342642], \
          [5e-4, 0.0012, 3.0853e-5, 25912430], \
          [6e-4, 0.0017, 3.6771e-5, 21381013], \
          [7e-4, 0.0023, 4.2612e-5, 18295367], \
          [8e-4, 0.0031, 4.9589e-5, 16132123], \
          [9e-4, 0.0037, 5.5047e-5, 14390035]]

def run(input_folder, output_folder, filename, header_line):

    print_epoch= 1000000

    with open(input_folder + filename) as file:
        print(filename + ' ...')
        all_lines = file.readlines()

    print(len(all_lines)/4)
    outstream= open(output_folder + filename, 'w+')
    outstream.write(' '.join([str(elt) for elt in header_line]) + '\n')
    for line_num in range(len(all_lines)/4):
        if (not line_num % print_epoch):
            print(line_num)
        synx= []
        errx= []
        synz= []
        errz= []
        for i in range(4):
            xz_str=  ''.join(all_lines[4*line_num + i].split('\t')).strip()
            synx.append(xz_str[0:3])
            synz.append(xz_str[3:6])
            errx.append(xz_str[6:13])
            errz.append(xz_str[13:20])
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
    for filename in os.listdir(sys.argv[1]):
        run(sys.argv[1], sys.argv[2], filename, headers[counter])
        sys.stdout.flush()
        counter+=1

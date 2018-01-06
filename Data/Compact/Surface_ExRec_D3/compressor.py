#
# Author: Pooya Ronagh (2017)
# All rights reserved.
#
# This script takes input data from Christopher Chamberland's Matlab code
# and strips spaces, leaving the distance 3 surface code data in the format
# synX1synX2synX3 errX1errX2errX3 synZ1synZ2synZ3 errZ1errZ2errZ3
# for every iteration. 
# Ever line contains: 12bits + ' ' + 12bits + ' ' + 27bits + ' ' + 27bits
# If the entire matrix is all zeros, it is excluded.
# 

from __future__ import print_function, division
from builtins import range
import numpy as np
import sys
import os
import json

headers= [[1e-4, 9.60e-5, 3.0982e-6, 1e+7], \
          [2e-4, 3.87e-4, 6.2197e-6, 1e+7], \
          [3e-4, 8.67e-4, 9.3072e-6, 1e+7]]

def run(input_folder, output_folder, filename, header_line):

    print_epoch= 100000

    with open(input_folder + filename) as file:
        print(filename + ' ...')
        all_lines = file.readlines()

    print(len(all_lines)/6)
    outstream= open(output_folder + filename, 'w+')
    outstream.write(' '.join([str(elt) for elt in header_line]) + '\n')
    for line_num in range(len(all_lines)/6):
        if (not line_num % print_epoch):
            print(line_num)
        synx= []
        errx= []
        synz= []
        errz= []
        for i in range(6):
            xz_str=  ''.join(all_lines[6*line_num + i].split('\t')).strip()
            synx.append(xz_str[0:4])
            synz.append(xz_str[4:8])
            errx.append(xz_str[8:17])
            errz.append(xz_str[17:26])
        result= ' '.join(synx) + ' ' + ' '.join(errx) \
        + ' ' + ' '.join(synz) + ' ' + ' '.join(errz)
        if '1' in result:
            outstream.write(result + '\n')
        # else:
        #     print('Warning at line ' + str(line_num))
        #     print(result)

    outstream.close()

if __name__ == '__main__':

    counter= 0
    for filename in os.listdir(sys.argv[1]):
        run(sys.argv[1], sys.argv[2], filename, headers[counter])
        sys.stdout.flush()
        counter+=1

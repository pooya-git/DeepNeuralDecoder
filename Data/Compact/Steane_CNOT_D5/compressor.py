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

# headers= [[1e-4, 6e-7, 4.7321e-7, 43776590], \
#           [2e-4, 3.6e-6, 1.3605e-6, 22523070], \
#           [3e-4, 1.5e-5, 2.9393e-6, 15307669], \
#           [4e-4, 3.41e-5, 4.3223e-6, 11718547], \
#           [5e-4, 6.17e-5, 6.0904e-6, 9580680], \
#           [6e-4, 1.097e-4, 8.1114e-6, 8172870], \
#           [7e-4, 1.818e-4, 1.0495e-5, 7134426], \
#           [8e-4, 2.750e-4, 1.3226e-5, 6365382], \
#           [9e-4, 3.645e-4, 1.4801e-5, 5817385]]

# headers= [[1e-3, 5.142e-4, 1.8163e-5, 5353460], \
#           [2e-3, 0.0038, 5.0043e-5, 3282450]]

headers= [[1.5e-3, 0.0016, 3.2240e-5, 3962882]]

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
            synx.append(xz_str[0:9])
            synz.append(xz_str[9:18])
            errx.append(xz_str[18:37])
            errz.append(xz_str[37:56])
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

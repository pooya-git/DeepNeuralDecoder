#
# Author: Pooya Ronagh (2017)
# All rights reserved.
#
# This script takes input data from Christopher Chamberland's Matlab code
# and strips X syndrome and error data. If there is any nonzero bit in X data
# it prints a lines of the form 
# (SynX1, SynX2, SynX3, SynX4) + ' ' + (ErrX3, ErrX4)
#

import numpy as np
import sys
import os
import json

# headers= [[1e-4, 4e-7, 2.7321e-7, 204354671],\
#           [2e-4, 4.2e-6, 1.4441e-6, 104384559],\
#           [3e-4, 1.77e-5, 3.0686e-6, 71033912],\
#           [4e-4, 3.81e-5, 4.6726e-6, 54564247], \
#           [5e-4, 7.03e-5, 6.2991e-6, 44618017], \
#           [6e-4, 1.264e-4, 8.1169e-6, 38153018], \
#           [7e-4, 1.968e-4, 1.0443e-5, 33392932], \
#           [8e-4, 3.023e-4, 1.297e-5, 29804533], \
#           [9e-4, 4.128e-4, 1.5251e-5, 27319830]]
   
headers= [[1e-3, 5.648e-4, 1.7716e-5, 25136174],\
          [1.5e-3, 0.0019, 3.3233e-5, 18778543],\
          [2e-3, 0.0043, 5.1170e-5, 15655645]]
    
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

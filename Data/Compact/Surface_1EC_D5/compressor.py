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

# Some older version had this data!
# headers= [[1e-4, 4e-6, 6.3245e-7, 43776590], \
#           [2e-4, 2.45e-5, 1.5652e-6, 22523070], \
#           [3e-4, 7.2e-5, 2.6832e-6, 15307669], \
#           [4e-4, 1.685e-4, 4.1045e-6, 11718547], \
#           [5e-4, 3.094e-4, 5.5615e-6, 9580680], \
#           [6e-4, 5.161e-4, 7.1822e-6, 8172870], \
#           [7e-4, 8.012e-4, 8.9474e-6, 7134426], \
#           [8e-4, 0.0012, 1.0776e-5, 6365382], \
#           [9e-4, 0.0016, 1.2732e-5, 5817385]]

# headers= [[1e-4, 3.3e-6, 5.7446e-7, 43776590] \
            # [2e-4, 2.59e-5, 1.6093e-6, 22523070] \
            # [3e-4, 9.36e-5, 3.0593e-6, 15307669] \
            # [4e-4, 2.099e-4, 4.5810e-6, 11718547] \
            # [5e-4, 3.96e-4, 6.2916e-6, 9580680] \
            # [6e-4, 6.635e-4, 8.1428e-6, 8172870] \
            # [7e-4, 0.0010, 1.0228e-5, 7134426] \
            # [8e-4, 0.0015, 1.2418e-5, 6365382] \
            # [9e-4, 0.0021, 1.4644e-5, 5817385]]

headers= [[1e-3, 0.0029, 1.7062e-5, 4783965], \
          [2e-3, 0.0193, 4.3481e-5, 3024051]]

def run(syn_folder, err_folder, output_folder, filename, header_line):

    print_epoch= 100000

    with open(syn_folder + filename) as file:
        print(filename + ' ...')
        syn_lines = file.readlines()

    with open(err_folder + filename) as file:
        print(filename + ' ...')
        err_lines = file.readlines()

    assert(len(syn_lines)==len(err_lines))
    print(len(syn_lines)/6)
    outstream= open(output_folder + filename, 'w+')
    outstream.write(' '.join([str(elt) for elt in header_line]) + '\n')
    for line_num in range(len(syn_lines)/6):
        if (not line_num % print_epoch):
            print(line_num)
        synx= []
        errx= []
        synz= []
        errz= []
        for i in range(6):
            xz_syn_str=  ''.join(syn_lines[6*line_num + i].split('\t')).strip()
            xz_err_str=  ''.join(err_lines[6*line_num + i].split('\t')).strip()
            synx.append(xz_syn_str[0:12])
            synz.append(xz_syn_str[12:24])
            errx.append(xz_err_str[0:25])
            errz.append(xz_err_str[25:50])
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
        run(sys.argv[1], sys.argv[2], sys.argv[3], filename, headers[counter])
        sys.stdout.flush()
        counter+=1

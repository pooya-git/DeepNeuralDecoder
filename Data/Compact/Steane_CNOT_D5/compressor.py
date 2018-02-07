#
# Author: Pooya Ronagh (2017)
# All rights reserved.
#
# This script takes input data from Christopher Chamberland's Matlab code
# and strips X syndrome and error data. If there is any nonzero bit in X data
# it prints a lines of the form 
# (SynX1, SynX2, SynX3, SynX4) + ' ' + (ErrX3, ErrX4)
#

import sys, os

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

headers= [[1e-3, 5.14e-4, 1.8131e-5, 49347664], \
          [1.5e-3, 0.0017, 3.3308e-5, 36944726], \
          [2e-3, 0.0039, 5.0647e-5, 30939335], \
          [6e-4, 1.138e-4, 8.4495e-6, 75031606], \
          [7e-4, 1.8420e-4, 1.0761e-5, 65084866], \
          [8e-4, 2.6630e-4, 1.2715e-5, 58883571], \
          [9e-4, 3.679e-4, 1.5087e-5, 53765723]]

def run(syn_folder, err_folder, output_folder, filename, header_line):

    print_epoch= 1000000

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
            synx.append(xz_syn_str[0:9])
            synz.append(xz_syn_str[9:18])
        for i in range(2):
            xz_err_str=  ''.join(err_lines[2*line_num + i].split('\t')).strip()
            errx.append(xz_err_str[0:19])
            errz.append(xz_err_str[19:38])
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

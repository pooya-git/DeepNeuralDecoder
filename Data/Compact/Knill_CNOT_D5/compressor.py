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

# headers= [[1e-4, 4e-7, 2.7321e-7, 40920217],\
#           [2e-4, 4.2e-6, 1.4441e-6, 20829172],\
#           [3e-4, 1.77e-5, 3.0686e-6, 14317264],\
#           [4e-4, 3.81e-5, 4.6726e-6, 10899812], \
#           [5e-4, 7.03e-5, 6.2991e-6, 8953941], \
#           [6e-4, 1.264e-4, 8.1169e-6, 7579201], \
#           [7e-4, 1.968e-4, 1.0443e-5, 6694884], \
#           [8e-4, 3.023e-4, 1.297e-5, 5985305], \
#           [9e-4, 4.128e-4, 1.5251e-5, 5448664]]
    
# headers= [[1e-3, 5.648e-4, 1.7716e-5, 5035105],\
#           [1.5e-3, 0.0019, 3.3233e-5, 3777545],\
#           [2e-3, 0.0043, 5.1170e-5, 3138974]]
    
headers= [[1e-3, 5.791e-4, 1.8096e-5, 44849944], \
          [1.5e-3, 0.0019, 3.3634e-5, 34039570], \
          [2e-3, 0.0044, 5.1848e-5, 28848591], \
          [6e-4, 1.297e-4, 8.3831e-6, 67357657], \
          [7e-4, 2.142e-4, 1.0789e-5, 58674450], \
          [8e-4, 3.012e-4, 1.2926e-5, 53174114], \
          [9e-4, 4.382e-4, 1.5798e-5, 48661156]]

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

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

import sys, os


headers= [[3e-4, 9.36e-5, 3.0593e-6, 133443824], \
          [4e-4, 2.099e-4, 4.5810e-6, 102734148], \
          [5e-4, 3.96e-4, 6.2916e-6, 84332489], \
          [6e-4, 6.635e-4, 8.1428e-6, 72160925], \
          [7e-4, 0.0010, 1.0228e-5, 63397696], \
          [8e-4, 0.0015, 1.2418e-5, 56898453]]

## new data set.
# headers= [[5e-4, 3.96e-4, 6.2916e-6, 47857280], \
#           [6e-4, 6.635e-4, 8.1428e-6, 41865355], \
#           [7e-4, 0.0010, 1.0228e-5, 37644806], \
#           [8e-4, 0.0015, 1.2418e-5, 34525192]]
   
def run(syn_folder, err_folder, output_folder, filename, header_line):

    print_epoch= 1000000

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
            synx.append(xz_syn_str[0:12])
            synz.append(xz_syn_str[12:24])
        for i in range(6):
            xz_err_str=  ''.join(err_lines[6*line_num + i].split('\t')).strip()
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
    for filename in sorted(os.listdir(sys.argv[1])):
        run(sys.argv[1], sys.argv[2], sys.argv[3], filename, headers[counter])
        sys.stdout.flush()
        counter+=1

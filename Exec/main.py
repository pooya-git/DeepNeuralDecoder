# -------------------------------------------------------------------------
# 
#    Executed the circuit trainer for all noise models.
#
#    Copyright (C) 2017 Pooya Ronagh
# 
# ------------------------------------------------------------------------

import sys
import os
import json
from time import localtime, strftime

sys.path.insert(0, 'Trainer')
from ExRecCNOTFullLSTM import train
from ExRecCNOTFullLSTM import get_data

if __name__ == '__main__':

    with open(sys.argv[1]) as paramfile:
        param = json.load(paramfile)
    # with open(sys.argv[2]) as networkfile:
    #     network = json.load(networkfile)

    output= []
    datafolder= sys.argv[2]

    for filename in os.listdir(datafolder):
        # Read data and find how much null syndromes to assume for error_scale
        print("Reading data from " + filename)
        raw_data, p, lu_avg, lu_std, data_size = get_data(datafolder + filename)

        # Train over the raw data
        output.append(train(datafolder + filename, param,\
            raw_data, p, lu_avg, lu_std, data_size))

    outfilename = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    f = open('Reports/' + outfilename + '.json', 'w')
    f.write(json.dumps(output, indent=2))
    f.close()


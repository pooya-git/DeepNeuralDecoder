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

sys.path.insert(0, '../CircuitTrainer')
import ExRecCNOTLabLSTM as trainer


if __name__ == '__main__':

    with open(sys.argv[1]) as paramfile:
        param = json.load(paramfile)
    with open(sys.argv[2]) as networkfile:
        network = json.load(networkfile)

    output= []
    datafolder= sys.argv[3]

    for filename in os.listdir(datafolder):
        output.append(trainer.train(datafolder + filename, param, network))

    outfilename = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    f = open('../Reports/' + outfilename + '.json', 'w')
    f.write(json.dumps(output, indent=2))
    f.close()


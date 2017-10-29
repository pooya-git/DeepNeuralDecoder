# ------------------------------------------------------------------------------
# 
#    Deep learning utilities.
#
#    Copyright (C) 2017 Pooya Ronagh
# 
# ------------------------------------------------------------------------------

import numpy as np
from builtins import range

def y2indicator(y, width):
    
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, width)).astype(np.int32)
    for i in range(N):
        if (y[i] >= 0):
            ind[i, y[i]] = 1
    return ind

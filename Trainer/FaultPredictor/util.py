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
    ind = np.matrix(np.zeros((N, width))).astype(np.int8)
    for i in range(N):
        assert(y[i] >= 0)
        ind[i, y[i]] = 1
    return ind

def vec_to_index(vec):

    return np.dot(vec, np.matrix(2**np.arange(vec.shape[1])[::-1]).transpose())

def perp(key):

    if (key=='X'):
        return 'Z'
    if (key=='Z'):
        return 'X'
    if (key=='errX3'):
        return 'errZ3'
    if (key=='errX4'):
        return 'errZ4'
    if (key=='errZ3'):
        return 'errX3'
    if (key=='errZ4'):
        return 'errX4'
    print('Error: Unrecognized key for perp module!')

# ---------------------------------------------------------------------
#
# This module contains common functions used in different programs
#
# ---------------------------------------------------------------------
import subprocess as sp
import numpy as np

# ---------------------------------------------------------------------
#
# Quantize a matrix by passing a divisor in
# Returns a quantized matrix with the same type as the input matrix
#
# ---------------------------------------------------------------------
def quantize(A, div, retType = None):
    if retType is None:
        retType = A.dtype
    ret = np.multiply(A,1/div)
    ret = np.trunc(ret)
    ret = np.multiply(ret,div)
    ret = np.asarray(ret, retType)
    return ret

# ---------------------------------------------------------------------
#
# MAIN
#
# ---------------------------------------------------------------------
if __name__ == "__main__":

    sp.call('cls', shell = True)

    A = np.array(range(0,100)).reshape(10,10)
    print('Matrix A:')
    print(A)
    print('')

    print('Testing quantization:')
    print(quantize(A, 2))

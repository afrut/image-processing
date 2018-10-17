# ---------------------------------------------------------------------
#
# Different ways to calculate the Discrete Cosine Transform of a Matrix
#
# ---------------------------------------------------------------------
import subprocess as sp
import numpy as np
import cv2
from scipy import fftpack as fp

# ---------------------------------------------------------------------
#
# DCT function through a transform matrix
#
# ---------------------------------------------------------------------
def dct(A, inv = False):
    dim = (A.shape[0], A.shape[1])

    # check if the matrix is square
    if dim[0] != dim[1]:
        print('Matrix is not square. Cannot compute DCT')
        return -1

    # preallocate output matrix
    T = np.zeros([dim[0],dim[0]]);

    # build the transform matrix
    for cntX in range(0,dim[0]):
        for cntY in range(0,dim[1]):
            if cntX == 0:
                T[cntX,cntY] = 1 / np.sqrt(dim[0]);
            else:
                T[cntX,cntY] = np.sqrt(2 / dim[0]) * \
                    np.cos((np.pi * ((2 * cntY) + 1) * cntX) / (2 * dim[0]))

    if inv == False:
        # perform the DCT transformation through matrix operations
        ret = np.matmul(T, A);
        ret = np.matmul(ret, np.transpose(T));
    else:
        # perform the inverse DCT transformation
        ret = np.matmul(np.transpose(T), A)
        ret = np.matmul(ret, T)
    return ret

# ---------------------------------------------------------------------
#
# Another way to calculate the DCT of a matrix A
#
# ---------------------------------------------------------------------
def dct2(A, inv = False):
    # get dimensions of input matrix
    dim = (A.shape[0], A.shape[1])

    # check if the matrix is square
    if dim[0] != dim[1]:
        print('Matrix is not square. Cannot compute DCT')
        return -1

    # internal normalization function
    def alpha(n):
        if(n == 0):
            return np.sqrt(1 / dim[0])
        else:
            return np.sqrt(2 /dim[0])

    # preallocate output matrix
    B = np.zeros([dim[0],dim[1]])

    if inv:
        # calculate the inverse DCT of A
        for x in range(0, dim[0]):
            for y in range(0, dim[0]):
                term = 0
                for u in range(0, dim[0]):
                    for v in range(0, dim[0]):
                        term = term + \
                               A[u,v] * alpha(u) * alpha(v) * \
                               np.cos(((2 * x) + 1) * u * np.pi / (2 * dim[0])) * \
                               np.cos(((2 * y) + 1) * v * np.pi / (2 * dim[0]))
                B[x,y] = term
    else:
        # calculate the DCT of A
        for u in range(0,dim[0]):
            for v in range(0,dim[1]):
                term = 0
                for x in range(0,dim[0]):
                    for y in range(0,dim[1]):
                        term = term + \
                               A[x,y] * alpha(u) * alpha(v) * \
                               np.cos(((2 * x) + 1) * u * np.pi / (2 * dim[0])) * \
                               np.cos(((2 * y) + 1) * v * np.pi / (2 * dim[0]))
                B[u,v] = term

    return B

# ---------------------------------------------------------------------
#
# Compare results of different ways to calculate DCT
#
# ---------------------------------------------------------------------
if __name__ == "__main__":
    
    sp.call('cls', shell = True)

    np.set_printoptions(precision = 2, suppress = True)

    A = np.array([[140,144,147,140,140,155,179,175]
                 ,[144,152,140,147,140,148,167,179]
                 ,[152,155,136,167,163,162,152,172]
                 ,[168,145,156,160,152,155,136,160]
                 ,[162,148,156,148,140,136,147,162]
                 ,[147,167,140,155,155,140,136,162]
                 ,[136,156,123,167,162,144,140,147]
                 ,[148,155,136,155,152,147,147,136]])
    B1 = dct(A)
    B2 = dct2(A)
    B3 = cv2.dct(A.astype('float32'))
    B4 = np.zeros([8,8])
    B5 = fp.dct(fp.dct(A.T, norm = 'ortho').T, norm = 'ortho')
    print('Input Matrix:')
    print(A)
    print('--------------------------------------------------')
    print('DCT through a transform matrix output and inverse:')
    print(B1)
    print(dct(B1,True))
    print('--------------------------------------------------')
    print('DCT through formulas output and inverse:')
    print(B2)
    print(dct2(B2,True))
    print('--------------------------------------------------')
    print('DCT through opencv function output and inverse:')
    print(B3)
    print(cv2.dct(B3.astype('float32'), B4, cv2.DCT_INVERSE))
    print('--------------------------------------------------')
    print('DCT through scipy.fftpack.dct output and inverse:')
    print(B5)
    print(fp.idct(fp.idct(B5.T, norm = 'ortho').T, norm = 'ortho'))

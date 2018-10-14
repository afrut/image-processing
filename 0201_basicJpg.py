#TODO: use opencv libraries
import subprocess as sp
import numpy as np
import cv2
import math

sp.call('cls', shell=True)

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
# Compare DCT functions with opencv DCT function
#
# ---------------------------------------------------------------------
def testDct():
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
    print('Input Matrix:')
    print(A)
    print('--------------------------------------------------')
    print('DCT1 output and inverse:')
    print(B1)
    print(dct(B1,True))
    print('--------------------------------------------------')
    print('DCT2 output and inverse:')
    print(B2)
    print(dct2(B2,True))
    print('--------------------------------------------------')
    print('opencv DCT output and inverse:')
    print(B3)
    print(cv2.dct(B3.astype('float32'), B4, cv2.DCT_INVERSE))
    print('--------------------------------------------------')
    print('Errors in DCT calculations:')
    print(np.trunc(B1 - B3))
    print(np.trunc(B2 - B3))

# ---------------------------------------------------------------------
#
# Quantization functions
#
# ---------------------------------------------------------------------
def basicQuantize(A,div):
    return np.multiply(np.trunc(np.multiply(A,1/div)),div)

# ---------------------------------------------------------------------
#
# jpeg function
#
# ---------------------------------------------------------------------
def jpeg(img, div, display8x8 = False):

    # begin raster scan
    stop = False
    display = False
    display8x8 = False
    matSize = 8         # size of matrix blocks to perform dct
    imgCopy = img.copy()
    for rowStart in range(0,512,matSize):
        for colStart in range(0,512,matSize):
            # take a matSize x matSize block
            imgSlice = img[rowStart:rowStart + matSize
                          ,colStart:colStart + matSize]

            # compute the discrete cosine transform of the block
            B = cv2.dct(imgSlice.astype('float32'))

            # quantize the block of DCT coefficients
            Bprime = basicQuantize(B,div)

            # calculate the inverse dct
            B = cv2.dct(Bprime.astype('float32'), B, cv2.DCT_INVERSE)

            # replace the quantized blocks in the image's copy
            imgCopy[rowStart:rowStart + matSize
                   ,colStart:colStart + matSize] = B

            # display the 8 x 8 blocks for human inspection
            if display8x8:

                # display an image with another resolution on a resizable window
                winName = 'Original'
                cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
                cv2.moveWindow(winName, winPosX, winPosY)
                cv2.imshow(winName,imgSlice)
                cv2.resizeWindow(winName, winWidth, winHeight)

                # increment next position of windows
                #winPosX = winPosX + winWidth

                # start plotting on another row
                if winPosX > 1430:
                    winPosY = winPosY + winHeight + 32
                    winPosX = 50

                # wait for a keypress from the user
                k = cv2.waitKey(0)
                if(k == 27):
                    stop = True

                # check if loop should be stopped
                if stop:
                    display8x8 = False

                # close all windows
                cv2.destroyAllWindows()

    return imgCopy

# ---------------------------------------------------------------------
#
# simple function to show an image at a certain location in the screen
#
# ---------------------------------------------------------------------
def showImg(img, winName, winPosX, winPosY):
    winName = 'div = 2**' + str(pwr)
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.moveWindow(winName, winPosX, winPosY)
    cv2.imshow(winName, img)
    cv2.resizeWindow(winName, winWidth, winHeight)

# ---------------------------------------------------------------------
#
# MAIN
#
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sp.call('cls', shell = True)

    # load a color image in grayscale
    img = cv2.imread('./test images/peppers_gray.tif', 0)

    # initialize x and y position of windows
    winPosY = 0         # y-coordinate of next window to plot
    winPosX = 60        # x-coordinate of next window to plot
    winWidth = 300      # width of window
    winHeight = 300     # height of window

    # ----------------------------------------
    # for checking if DCT is computed correctly
    # ----------------------------------------
    #testDct()

    # ----------------------------------------
    # jpeg-type compression
    # ----------------------------------------
    print('Performing jpeg-type compression with DCT and quantization')
    for pwr in range(0,12):
        # divisor for quantization
        div = 2**pwr

        # perform jpeg quantization on the image
        ret = jpeg(img, div)

        # display an image with another resolution on a resizable window
        showImg(ret, 'div = 2**' + str(pwr), winPosX, winPosY)

        # start plotting on another row
        if winPosX > 1430:
            winPosY = winPosY + winHeight + 32
            winPosX = 60
        else:
            # increment next position of windows
            winPosX = winPosX + winWidth

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ----------------------------------------
    # comparison of quantization only and quantization with DCT
    # ----------------------------------------
    # initialize x and y position of windows
    winPosY = 0         # y-coordinate of next window to plot
    winPosX = 60        # x-coordinate of next window to plot
    winWidth = 300      # width of window
    winHeight = 300     # height of window

    sp.call('cls', shell = True)
    print('Performing quantization without DCT')
    for pwr in range(0,6):
        # divisor for quantization
        div = 2**pwr

        # perform jpeg quantization on the image
        retJpeg = jpeg(img, div)

        # perform quantization on the image
        # TODO: get basic quantization to work here
        retQuant = basicQuantize(img, div)       

        # display an image with another resolution on a resizable window
        showImg(retJpeg, 'JPEG div = 2**' + str(pwr), winPosX, winPosY)
        showImg(img, 'QUANT div = 2**' + str(pwr), winPosX, winPosY + winHeight)
        print(retQuant)

        # start plotting on another row
        if winPosX > 1430:
            winPosY = winPosY + winHeight + 32
            winPosX = 60
        else:
            # increment next position of windows
            winPosX = winPosX + winWidth

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

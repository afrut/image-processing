# exec(open('0103_rotate.py').read())
#TODO: see if opencv has a rotation function
import subprocess as sp
import numpy as np
import cv2
import math
import classes

sp.call('cls', shell=True)

# ---------------------------------------------------------------------
#
# Averaging function
#
# ---------------------------------------------------------------------
def imRotate(img, theta, ctr):
    # get dimensions of image
    numRows = img.shape[0]
    numCols = img.shape[1]

    # preallocate the output image
    imgCopy = np.zeros( [512,512], dtype=np.uint8 )

    brk = False

    # loop through every pixel in the output image and check if each
    # pixel maps to a pixel in the input image
    for i in range(0,numRows):
        for j in range(0,numCols):
            x = int(math.cos(theta) * (i - ctr[0]) - \
                    math.sin(theta) * (j - ctr[1]) + ctr[0])
            y = int(math.sin(theta) * (i - ctr[0]) +
                    math.cos(theta) * (j - ctr[1]) + ctr[1])

            # check if the current output pixel has a corresponding
            # input pixel
            if x >= 0 and x < 512 and y >= 0 and y < 512:
                # set the value of the output pixel to the corresponding
                # input pixel
                imgCopy.itemset( (i,j)  
                               , img.item(x,y) )

    return imgCopy

# ---------------------------------------------------------------------
#
# MAIN
#
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # load a color image in grayscale
    img = cv2.imread('./test images/peppers_gray.tif', 0)

    # image display manager
    idm = classes.ImageDisplayManager()

    # display original image
    idm.add(img, 'Original')

    # define center around which rotation occurs
    ctr = ( math.trunc(img.shape[0] / 2) + 1
          , math.trunc(img.shape[1] / 2) + 1 )

    # for all powers of 2 from 0 to 8
    for n in range(0,9):
        imgCopy = imRotate( img, n * math.pi / 4, ctr)

        # display an image with another resolution on a resizable window
        idm.add(imgCopy, 'n = ' + str(n) + '* pi')

    # show all images
    idm.show()

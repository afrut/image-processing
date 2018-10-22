#TODO: see if there is an opencv function for intensity quantization
import subprocess as sp
import numpy as np
import cv2
import math
import time

import commonFunc as cf

# ---------------------------------------------------------------------
#
# MAIN
#
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # clear screen
    sp.call('cls', shell=True)

    # initialize ImageDisplayManager
    idm = cf.ImageDisplayManager()

    # load a color image in grayscale
    img = cv2.imread('./test images/peppers_gray.tif', 0)
    imgCopy = img.copy()

    # for all powers of 2 from 0 to 8
    for pwr in range(0, 8):

        # restore the image
        img = imgCopy.copy()

        # quantize the intensity levels
        img = cf.quantize(img, 2**pwr)

        # add image to be displayed later
        idm.add(img, '2 ** {}'.format(pwr))

    # show all images
    idm.show()

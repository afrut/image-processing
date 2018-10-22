#TODO: see if opencv has a blurring or averaging function
import subprocess as sp
import numpy as np
import cv2
import math

import commonFunc as cf

# ---------------------------------------------------------------------
#
# MAIN
#
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sp.call('cls', shell=True)

    # instantiate ImageDisplayManager
    idm = cf.ImageDisplayManager()

    # load a color image in grayscale
    img = cv2.imread('./test images/peppers_gray.tif', 0)

    # perform averaging for different nxn moving windows
    for n in range(3,33,2):
        imgCopy = cf.average(img,n)

        # display an image with another resolution on a resizable window
        idm.add(imgCopy, 'n = {}'.format(n))

    # show all images
    idm.show()

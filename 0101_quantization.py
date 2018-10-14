#TODO: see if there is an opencv function for intensity quantization
import subprocess as sp
import numpy as np
import cv2
import math

sp.call('cls', shell=True)

# ---------------------------------------------------------------------
#
# Quantization function by passing in divisor
#
# ---------------------------------------------------------------------
def quantizeDiv(img, div):
    
    # get number of rows and columns of image
    numRows = img.shape[0]
    numCols = img.shape[1]

    # loop through all pixels
    for i in range(0,numRows):
        for j in range(0,numCols):
            
            # quantize based on divisor passed in
            img.itemset((i,j),math.trunc(img.item(i,j) / div) * div)

# ---------------------------------------------------------------------
#
# Quantization function using numpy matrix functions
#
# ---------------------------------------------------------------------
def matQuantize(A,div):
    ret = np.multiply(A,1/div)
    ret = np.trunc(ret)
    ret = np.multiply(ret,div)
    ret = np.asarray(ret,dtype = np.uint8)
    return ret

# ---------------------------------------------------------------------
#
# MAIN
#
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # load a color image in grayscale
    img = cv2.imread('./test images/peppers_gray.tif', 0)
    imgCopy = img.copy()

    # initialize x and y position of windows
    winPosY = 0         # y-coordinate of next window to plot
    winPosX = 50        # x-coordinate of next window to plot
    winWidth = 345      # width of window
    winHeight = 345     # height of window

    # for all powers of 2 from 0 to 8
    for pwr in range(0, 8):

        # restore the image
        img = imgCopy.copy()

        # quantize the intensity levels
        #quantizeDiv(img, 2 ** pwr)
        img = matQuantize(img, 2**pwr)

        # display an image with another resolution on a resizable window
        winName = '2 ** ' + str(pwr)      # name of the current window
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.moveWindow(winName, winPosX, winPosY)
        cv2.imshow(winName,img)
        cv2.resizeWindow(winName, winWidth, winHeight)

        # increment next position of windows
        winPosX = winPosX + winWidth

        # start plotting on another row
        if winPosX > 1430:
            winPosY = winPosY + winHeight + 32
            winPosX = 50

    # wait for a keypress from the user
    k = cv2.waitKey(0)

    # close all windows
    cv2.destroyAllWindows()

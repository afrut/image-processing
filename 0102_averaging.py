#TODO: see if opencv has a blurring or averaging function
import subprocess as sp
import numpy as np
import cv2
import math

sp.call('cls', shell=True)

# ---------------------------------------------------------------------
#
# Averaging function
#
# ---------------------------------------------------------------------
def imAverage(img,n):
    # copy the image
    imgCopy = np.copy(img)

    # get dimensions of image
    numRows = img.shape[0]
    numCols = img.shape[1]

    brk = False

    # loop through every pixel
    for i in range(0,numRows):
        for j in range(0,numCols):
            # calculate the number of indices to change
            delta = int((n-1)/2)

            # calculate start and end positions of x and y coordinates
            xStart = max(i-delta,0)
            xEnd = min(i+delta+1,numRows-1)
            yStart = max(j-delta,0)
            yEnd = min(j+delta+1,numCols-1)

            # get a slice of the image
            px = img[xStart:xEnd,yStart:yEnd]

            # replace the current pixel with neighboring average
            imgCopy.itemset((i,j),np.average(px))

    return imgCopy

# ---------------------------------------------------------------------
#
# MAIN
#
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # load a color image in grayscale
    img = cv2.imread('./test images/peppers_gray.tif', 0)

    # initialize x and y position of windows
    winPosY = 0         # y-coordinate of next window to plot
    winPosX = 50        # x-coordinate of next window to plot
    winWidth = 345      # width of window
    winHeight = 345     # height of window
    
    # for all powers of 2 from 0 to 8
    for n in range(3,33,2):
        imgCopy = imAverage(img,n)

        # display an image with another resolution on a resizable window
        winName = 'n=' + str(n)     # name of the current window
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.moveWindow(winName, winPosX, winPosY)
        cv2.imshow(winName,imgCopy)
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

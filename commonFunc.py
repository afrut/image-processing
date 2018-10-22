# ---------------------------------------------------------------------
#
# This module contains common functions used in different programs
#
# ---------------------------------------------------------------------
import subprocess as sp
import numpy as np
import time
import cv2
import math

# ---------------------------------------------------------------------
#
# A simple class to handle the management of displaying images properly
# on screen; only designed to have one instance
#
# ---------------------------------------------------------------------
class ImageDisplayManager:
    images = list()     # list of all images to display

    # ----------------------------------------
    # to be called by client code to add images to display
    # ----------------------------------------
    def add(self, img, title):
        ImageDisplayManager.images.append((img, title))

    # ----------------------------------------
    # to be called by client code to add display all images
    # ----------------------------------------
    def show(self):
        # initialize positions
        x = 60
        y = 0
        width = 300
        height = 300

        # tpl[0] is the image to be displayed
        # tpl[1] is the title for the image
        for tpl in ImageDisplayManager.images:
            img = tpl[0].copy()
            title = tpl[1]

            # call the appropriate opencv functions to display the images
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.moveWindow(title, x, y)
            cv2.imshow(title, img)
            cv2.resizeWindow(title, width, height)

            # start plotting on another row
            if x > 1430:
                y = y + height + 32
                x = 60
            else:
                # increment next position of windows
                x = x + width

        # wait for user input then destroy all windows
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------
#
# Quantize a matrix by passing a divisor in
# Returns a quantized matrix with the same type as the input matrix
#
# ---------------------------------------------------------------------
def quantize(A, div, retType = None):

    # return the same matrix type as the input
    if retType is None:
        retType = A.dtype

    # divide, truncate, multiply
    ret = np.multiply(A,1/div)
    ret = np.trunc(ret)
    ret = np.multiply(ret,div)

    # return as appropriate type
    ret = np.asarray(ret, retType)
    return ret

# ---------------------------------------------------------------------
#
# Averages an nxn neighborhood of pixels and replaces the center
# pixel with the average
#
# ---------------------------------------------------------------------
def neighborAverage(img,n):
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
            delta = int((n - 1)/2)

            # calculate start and end positions of x and y coordinates
            xStart = max(i - delta,0)
            xEnd = min(i + delta + 1, numRows - 1)
            yStart = max(j - delta,0)
            yEnd = min(j + delta + 1,numCols - 1)

            # get a slice of the image
            px = img[xStart:xEnd,yStart:yEnd]

            # replace the current pixel with neighboring average
            imgCopy.itemset((i,j),np.average(px))

    return imgCopy

# ---------------------------------------------------------------------
#
# Rotates an image
#
# ---------------------------------------------------------------------
def rotate(img, theta, ctr = None):

    # define center around which rotation occurs
    if ctr is None:
        ctr = ( math.trunc(img.shape[0] / 2) + 1
              , math.trunc(img.shape[1] / 2) + 1 )

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

    # clear screen
    sp.call('cls', shell = True)

    # initialize ImageDisplayManager
    idm = ImageDisplayManager()

    # load a color image in grayscale
    img = cv2.imread('./test images/peppers_gray.tif', 0)
    idm.add(img, 'Original image')

    # perform quantization, measure time taken, and add to display list
    div = 2 # divisor used for quantization
    tmStart = time.time()
    imgQuantized = quantize(img, 2)
    tmEnd = time.time()
    tmElapsed = round((tmEnd - tmStart) * 1000, 2)
    print('Quantization time elapsed = {} ms'.format(tmElapsed))
    idm.add(imgQuantized, 'Quantized image, div = {}'.format(div))

    # perform averaging of pixels based on neighboring pixels
    n = 3   # the size of the sliding window with which to average, needs to be odd
    tmStart = time.time()
    imgAveraged = neighborAverage(img, n)
    tmEnd = time.time()
    tmElapsed = round((tmEnd - tmStart) * 1000, 2)
    print('Averaging time elapsed = {} ms'.format(tmElapsed))
    idm.add(imgAveraged, 'Averaged image, n = {}'.format(n))

    # perform averaging of pixels based on neighboring pixels
    tmStart = time.time()
    imgRotated = rotate(img, math.pi / 4)
    tmEnd = time.time()
    tmElapsed = round((tmEnd - tmStart) * 1000, 2)
    print('Rotation time elapsed = {} ms'.format(tmElapsed))
    idm.add(imgRotated, 'Rotated image')

    # show all images
    idm.show()


# exec(open('0201_jpeg.py').read())
import subprocess as sp
import numpy as np
import cv2
import math
import commonFunc as cf
import classes

sp.call('cls', shell=True)

# ---------------------------------------------------------------------
#
# jpeg function
#
# ---------------------------------------------------------------------
def jpeg(img, div, fft = False, display8x8 = False):
    stop = False
    display = False
    matSize = 8             # size of matrix blocks to perform dct
    imgCopy = img.copy()
    idm = classes.ImageDisplayManager()

    # begin scanning
    for rowStart in range(0,512,matSize):
        for colStart in range(0,512,matSize):
            # take a matSize x matSize block
            imgSlice = img[rowStart:rowStart + matSize
                          ,colStart:colStart + matSize]

            if(not fft):
                # compute the discrete cosine transform of the block
                B = cv2.dct(imgSlice.astype('float32'))

                # quantize the block of DCT coefficients
                Bprime = cf.quantize(B,div)

                # calculate the inverse dct
                B = cv2.dct(Bprime.astype('float32'), B, cv2.DCT_INVERSE)
            else:
                # compute the discrete cosine transform of the block
                B = np.fft.fft2(imgSlice)

                # quantize the block of DCT coefficients
                Bprime = cf.quantize(B,div)

                # calculate the inverse dct
                B = np.fft.ifft2(Bprime)
                
            # replace the quantized blocks in the image's copy
            imgCopy[rowStart:rowStart + matSize
                   ,colStart:colStart + matSize] = B

            # display the 8 x 8 blocks for human inspection
            if display8x8:

                # display an image with another resolution on a resizable window
                idm.showImg(imgSlice, '8 x 8')

    return imgCopy





# ---------------------------------------------------------------------
#
# MAIN
#
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sp.call('cls', shell = True)

    # load a color image in grayscale
    img = cv2.imread('./test images/peppers_gray.tif', 0)

    # ----------------------------------------
    # jpeg-type compression
    # ----------------------------------------
    idm = classes.ImageDisplayManager()
    for pwr in range(0,12):
        # divisor for quantization
        div = 2**pwr

        # perform jpeg quantization on the image
        ret = jpeg(img, div, fft = False, display8x8 = False)

        # add each of the figures to be displayed
        idm.add(ret, 'div = 2**' + str(pwr))
    idm.show()

    # ----------------------------------------
    # comparison of quantization only and quantization with DCT
    # ----------------------------------------
    sp.call('cls', shell = True)
    idm.init()
    for pwr in range(0,12,2):
        # divisor for quantization
        div = 2**pwr

        # perform jpeg quantization on the image
        retJpeg = jpeg(img, div)

        # perform quantization on the image
        retQuant = cf.quantize(img, div)       

        # display an image with another resolution on a resizable window
        idm.add(retJpeg, 'JPEG div = 2**' + str(pwr))
        idm.add(retQuant, 'QUANT div = 2**' + str(pwr))
    idm.show()

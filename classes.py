# ---------------------------------------------------------------------
#
# This module contains common classes used in different programs
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
    
    # constructor
    def __init__(self
                ,x0 = 60
                ,y0 = 0
                ,winWidth = 300
                ,winHeight = 300
                ,xMax = 1730):
        # x and y start positions for this instance
        self.x0 = x0
        self.y0 = y0

        # height and width of window
        self.winWidth = winWidth
        self.winHeight = winHeight

        # x and y position of the next window to be displayed
        self.y = self.y0
        self.x = self.x0

        # boolean to determine if showing an individual image is enabled
        self.showOne = True

        self.xMax = xMax        # maximum x position; start another row
        self.images = list()    # list of images to display

    # ----------------------------------------
    # to reinitialize this instance
    # ----------------------------------------
    def init(self):
        self.x = self.x0
        self.y = self.y0
        self.images = list()
        self.showOne = True

    # ----------------------------------------
    # to be called by client code to add images to display
    # ----------------------------------------
    def add(self, img, title):
        self.images.append((img.copy(), title))

    # ----------------------------------------
    # to be called by client code to display one image
    # ----------------------------------------
    def showImg(self, img, title):
        if self.showOne:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)               # create window
            cv2.moveWindow(title, self.x0, self.y0)                 # move window
            cv2.imshow(title, img)                                  # show image
            cv2.resizeWindow(title, self.winWidth, self.winHeight)  # resize window

            # wait for user keypress
            k = cv2.waitKey(0)
            if(k == 27):
                # user pressed escape
                self.showOne = False
            cv2.destroyAllWindows()

    # ----------------------------------------
    # to be called by client code to display all images
    # ----------------------------------------
    def show(self):
        # tpl[0] is the image to be displayed
        # tpl[1] is the title for the image
        for tpl in self.images:
            img = tpl[0]
            title = tpl[1]

            cv2.namedWindow(title, cv2.WINDOW_NORMAL)               # create window
            cv2.moveWindow(title, self.x, self.y)                   # move window
            cv2.imshow(title, img)                                  # show image
            cv2.resizeWindow(title, self.winWidth, self.winHeight)  # resize window

            # increment positions
            self.x = self.x + self.winWidth

            # check if another displaying of figures needs to increment
            # to the next row
            if self.x > self.xMax:
                self.y = self.y + self.winHeight + 32
                self.x = self.x0

        # wait for user input then destroy all windows and empty list
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.init()

# Built-int imports 
import os
import sys
import argparse

# External imports
import cv2 as cv
import numpy as np

# My own imports 
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

class HousePickUpColor:
    """
    Python class that has all the necessary methods to solve the third challenge
    of the EIA University's class of computer vision.
    This is a python class that contains methods to perform a color segmentation 
    using the HSV image format and to play modifying an image.
    """
    def __init__(self):
        self.__image_path = self.get_image_path("fachada1.png")
        self.__image = cv.imread(self.__image_path, 1)
        self.__rows, self.__cols = self.__image.shape[:2]
        self.hsv_image = cv.cvtColor(self.__image, cv.COLOR_RGB2HSV)

    def get_image_path(self, image_name):
        image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", image_name)
        return image_relative_path
    
    def nothing(self, x):
        """
        nothing function that is necessary for the trackbar event        
        """
        pass

    def create_trackbar(self, name, title):
        return cv.createTrackbar(
            name,
            title,
            0,
            255,
            self.nothing
        )

    def color_segmentation(self):
        """
        This is a method to make the color segmentation of the image called
        'fachada1.png', the segmentation will be using six different trackbars
        two per channel (H,S,V). 
        """
        cv.namedWindow("Segmentation parameters")
        self.create_trackbar("h-u", "Segmentation parameters")
        self.create_trackbar("h-l","Segmentation parameters")
        self.create_trackbar("s-u","Segmentation parameters")
        self.create_trackbar("s-l","Segmentation parameters")
        self.create_trackbar("v-u","Segmentation parameters")
        self.create_trackbar("v-l","Segmentation parameters")

        image = self.__image.copy()

        while True:
            var_h_upper = cv.getTrackbarPos("h-u", "Segmentation parameters")
            var_h_lower = cv.getTrackbarPos("h-l", "Segmentation parameters")
            var_s_upper = cv.getTrackbarPos("s-u", "Segmentation parameters")
            var_s_lower = cv.getTrackbarPos("s-l", "Segmentation parameters")
            var_v_upper = cv.getTrackbarPos("v-u", "Segmentation parameters")
            var_v_lower = cv.getTrackbarPos("v-l", "Segmentation parameters")

            lower = np.array([var_h_lower,var_s_lower,var_v_lower])
            upper = np.array([var_h_upper,var_s_upper,var_v_upper])

            bin_image = cv.inRange(self.hsv_image, lower, upper)
            cv.imshow("Segmentated image", bin_image)

            if (cv.waitKey(1) & 0xFF == ord('q')):
                break
        cv.destroyAllWindows()

    def click_mouse_callback(self, event, y, x, flags, param):
        if (event == cv.EVENT_LBUTTONUP):
            self.pixel_picked = self.color_palette[x,y]
            print(self.pixel_picked)

    def segmentation_condition(self, i , j):
        return (self.hsv_image[i,j,0] > 42 and self.hsv_image[i,j,0] < 92 and \
        self.hsv_image[i,j,1] > 54 and self.hsv_image[i,j,1] < 168 and \
        self.hsv_image[i,j,2] > 89 and self.hsv_image[i,j,2] < 245) and \
        not(self.pixel_picked[0] == 0) and not(self.pixel_picked[1] == 0) and \
        not(self.pixel_picked[2] == 0)

    def colorfull_house(self):
        """
        This is a method to play with the image 'fachada1.png, changing the 
        color of the house's front according to a different image of a color
        palette.'
        """
        image = self.__image.copy()
        palette_path = self.get_image_path("color_palette.png")
        self.color_palette = cv.resize(cv.imread(palette_path, 1),(598,245))
        self.pixel_picked = np.array([0,0,0])

        cv.namedWindow("Color Palette")
        cv.setMouseCallback("Color Palette", self.click_mouse_callback)
        cv.imshow("Color Palette", self.color_palette)

        while (True):
            for i in iter(range(self.__rows)):
                for j in iter(range(self.__cols)):
                    if self.segmentation_condition(i,j):
                        image[i,j] = self.pixel_picked
            
            cv.imshow("Colorfull House",image)
            if (cv.waitKey(1) & 0xFF == ord('q')):
                break

    

def main():
    """COMPUTER VISION - EIA UNIVERSITY
    Third challenge of the EIA University's computer vision class.
    Run this scripts in order to see Elkin Guerra's solucion 
    of this test. 
    """
    colorhouse = HousePickUpColor()
    colorhouse.colorfull_house()

if __name__=='__main__':
    sys.exit(main())
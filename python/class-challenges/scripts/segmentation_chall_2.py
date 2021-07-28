# Built-int imports 
import os
import sys
import math
import argparse

# External imports
import cv2 as cv
import numpy as np
from numpy.core.fromnumeric import size

# My own imports 
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()


class CoinSegmentation:
    """
    
    """
    def __init__(self):
        self.__coins_path = self.get_image_path("monedas.jpg")
        self.__image = cv.imread(self.__coins_path, 1)
        self.__gray_image = cv.imread(self.__coins_path, 0)
        self.__rows, self.__cols, self.__channels = self.__image.shape


    def get_image_path(self, image_name):
        image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", image_name)
        return image_relative_path

    def nothing(self, x):
        """
        nothing function that is necessary for the trackbar event        
        """
        pass

    def create_trackbar(self,name,title):
        return cv.createTrackbar(
            name,
            title,
            0,
            255,
            self.nothing
        )

    def color_segmentation(self):
        cv.namedWindow("Coins grup segmentation")
        self.create_trackbar("Blue-u", "Coins grup segmentation")
        self.create_trackbar("Blue-l","Coins grup segmentation")
        self.create_trackbar("Green-u","Coins grup segmentation")
        self.create_trackbar("Green-l","Coins grup segmentation")
        self.create_trackbar("Red-u","Coins grup segmentation")
        self.create_trackbar("Red-l","Coins grup segmentation")

        image = self.__image.copy()

        while(True):
            var_blue_upper = cv.getTrackbarPos("Blue-u", "Coins grup segmentation")
            var_blue_lower = cv.getTrackbarPos("Blue-l", "Coins grup segmentation")
            var_green_upper = cv.getTrackbarPos("Green-u", "Coins grup segmentation")
            var_green_lower = cv.getTrackbarPos("Green-l", "Coins grup segmentation")
            var_red_upper = cv.getTrackbarPos("Red-u", "Coins grup segmentation")
            var_red_lower = cv.getTrackbarPos("Red-l", "Coins grup segmentation")

            for i in iter(range(self.__rows)):
                for j in iter(range(self.__cols)):
                    if (var_blue_upper > image[i,j,0] and var_blue_lower < image[i,j,0] and \
                        var_green_upper > image[i,j,1] and var_green_lower < image[i,j,1] and \
                        var_red_upper > image[i,j,2] and var_red_lower < image[i,j,2]):
                        image[i,j] = 255
                    else:
                        image[i,j] = 0
                        
            cv.imshow("Coins grup segmentation", image)
            image = self.__image.copy()

            if (cv.waitKey(1) & 0xFF == ord('q')):
                break
        cv.destroyAllWindows()
    
    def group_coin_segmentation(self):
        image = self.__image.copy()

        for i in iter(range(self.__rows)):
            for j in iter(range(self.__cols)):
                if (image[i,j,0] > 50 and image[i,j,0] < 111 and \
                    image[i,j,1] > 101 and image[i,j,1] < 149 and \
                    image[i,j,2] > 137 and image[i,j,2] < 215):
                    image[i,j] = np.array([255,0,0])
                elif (image[i,j,0] > 23 and image[i,j,0] < 72 and \
                    image[i,j,1] > 64 and image[i,j,1] < 100 and \
                    image[i,j,2] > 75 and image[i,j,2] < 111):
                    image[i,j] = np.array([0,255,0])
                elif (image[i,j,0] > 28 and image[i,j,0] < 60 and \
                    image[i,j,1] > 71 and image[i,j,1] < 106 and \
                    image[i,j,2] > 119 and image[i,j,2] < 131):
                    image[i,j] = np.array([0,0,255])

        cv.imshow("Coin's groups", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def binarization(self):
        cv.namedWindow("Coins binarization")
        self.create_trackbar("Gray-u","Coins binarization")
        self.create_trackbar("Gray-l","Coins binarization")

        image = self.__gray_image.copy()

        while(True):
            var_gray_upper = cv.getTrackbarPos("Gray-u","Coins binarization")
            var_gray_lower = cv.getTrackbarPos("Gray-l","Coins binarization")
    
            for i in iter(range(self.__rows)):
                for j in iter(range(self.__cols)):
                    if (var_gray_upper > image[i,j] and var_gray_lower < image[i,j]):
                        image[i,j] = 255
                    else:
                        image[i,j] = 0
                        
            cv.imshow("Coins binarization", image)
            image = self.__gray_image.copy()

            if (cv.waitKey(1) & 0xFF == ord('q')):
                break
        cv.destroyAllWindows()
    
    def coin_count(self):
        image = self.__gray_image.copy()
        total_area = 0

        for i in iter(range(self.__rows)):
            for j in iter(range(self.__cols)):
                if (image[i,j] > 19 and image[i,j] < 171):
                    total_area += 1
        
        print(f"The total area is: {total_area} pixel^2")
        coin_size = total_area/8
        print(f"The average size of the coins is: {coin_size} pixel^2")

        other_image = cv.imread(
            self.get_image_path("monedas2.jpg"),
            0
        )

        other_img_area = 0
        for i in iter(range(self.__rows)):
            for j in iter(range(self.__cols)):
                if (other_image[i,j] > 19 and other_image[i,j] < 171):
                    other_img_area += 1

        print(f"Number of coins: {int(other_img_area/coin_size)}")




def main():
    coins = CoinSegmentation()

    coins.group_coin_segmentation()



if __name__=='__main__':
    sys.exit(main())
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


class CoinSegmentation:
    """
    
    """
    def __init__(self):
        self.__coins_path = self.get_image_path()
        self.__image = cv.imread(self.__coins_path, 1)
        self.__rows, self.__cols, self.__channels = self.__image.shape


    def get_image_path(self):
        image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", "monedas.jpg")
        return image_relative_path

    def nothing(self, x):
        """
        nothing function that is necessary for the trackbar event        
        """
        pass

    def group_coin_segmentation(self):
        cv.namedWindow("Coins grup segmentation")
        cv.createTrackbar(
            "Blue-upper",
            "Coins grup segmentation",
            0,
            255,
            self.nothing
        )
        cv.createTrackbar(
            "Blue-lower",
            "Coins grup segmentation",
            0,
            255,
            self.nothing
        )

        image = self.__image.copy()

        while(True):
            var_blue_upper = cv.getTrackbarPos("Blue-upper", "Coins grup segmentation")
            var_blue_lower = cv.getTrackbarPos("Blue-lower", "Coins grup segmentation")

            for i in iter(range(self.__rows)):
                for j in iter(range(self.__cols)):
                    if (var_blue_upper > image[i,j,0] and var_blue_lower < image[i,j,2]):
                        image[i,j] = 255
                    else:
                        image[i,j] = 0
                        
            cv.imshow("Coins grup segmentation", image)
            image = self.__image.copy()

            if (cv.waitKey(1) & 0xFF == ord('q')):
                break


def main():
    coins = CoinSegmentation()

    coins.group_coin_segmentation()



if __name__=='__main__':
    sys.exit(main())
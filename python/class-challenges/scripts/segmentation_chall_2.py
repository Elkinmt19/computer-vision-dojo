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

    def create_trackbar(self,name):
        return cv.createTrackbar(
            name,
            "Coins grup segmentation",
            0,
            255,
            self.nothing
        )

    def segmentation(self):
        cv.namedWindow("Coins grup segmentation")
        self.create_trackbar("Blue-u")
        self.create_trackbar("Blue-l")
        self.create_trackbar("Green-u")
        self.create_trackbar("Green-l")
        self.create_trackbar("Red-u")
        self.create_trackbar("Red-l")

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


def main():
    coins = CoinSegmentation()

    coins.group_coin_segmentation()



if __name__=='__main__':
    sys.exit(main())
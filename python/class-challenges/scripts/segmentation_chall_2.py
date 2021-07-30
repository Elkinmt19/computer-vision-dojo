# Built-int imports 
import os
import sys
import argparse

# External imports
import cv2 as cv
import numpy as np

# My own imports 
import get_path_assests_folder as gpaf
import custom_plot as cplt

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()


class CoinSegmentation:
    """
    Python class that has all the necessary methods to solve the second challenge
    of the EIA University's class of computer vision.
    This is a python class that contains methods to perform proper color
    segmentation and binarization on a coin image and also to play with other images.
    """
    def __init__(self):
        self.__coins_path = self.get_image_path("monedas.jpg")
        self.__image = cv.imread(self.__coins_path, 1)
        self.__gray_image = cv.imread(self.__coins_path, 0)
        self.__bars_gray_image = cv.imread(self.get_image_path("barras.png"),0)
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
        """
        This is a method to make the color segmentation of the image called
        'monedas.jpg', the segmentation will be using six different trackbars
        two per channel (blue, gree and red). 
        """
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
        """
        This is a method to make the binarization of the image called
        'monedas.jpg', this method will be using two different trackbars
        and the image is gonna be load in gray scale from the start.
        """
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
        
        print(f"Number of coins - image(monedas2.jpg): {int(other_img_area/coin_size)}")

    def coin_connect_components(self):
        """
        This is a method that gets the number of coins that the image 
        'monedas2.jpg' has using the information got from the processing of 
        the image 'monedas.jpg'.
        """
        image = self.__gray_image.copy()

        for i in iter(range(self.__rows)):
            for j in iter(range(self.__cols)):
                if (image[i,j] > 19 and image[i,j] < 171):
                    image[i,j] = 255
                else:
                    image[i,j] = 0
        
        num_labels, labels_im = cv.connectedComponents(image)
        print(f"Number of components: {num_labels}")

        label_hue = np.uint8(179*labels_im/np.max(labels_im))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv.merge((label_hue, blank_ch, blank_ch))

        # cvt to BGR for display
        labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue==0] = 0

        cv.imshow('Coins in groups', labeled_img)
        cv.waitKey()

    def count_bars(self):
        """
        This is a method that gets the number of bars that the image called
        'barras.png' has and also calculates the distantes between the bars.
        """
        image = self.__bars_gray_image.copy()

        row, cols = image.shape
        bar_refe = image[int(row/2),0]
        bar_number, dist_refe = (0,0)
        bar_dist = list()
        bar_number_plot = list()

        for i in iter(range(row)):
            for j in iter(range(cols)):
                if (image[i,j] > 60 and image[i,j] < 200):
                    image[i,j] = 150
                elif (image[i,j] > 200):
                    image[i,j] = 255
                else:
                    image[i,j] = 0

        for k in iter(range(cols)):
            dist_refe += 1
            if (image[int(row/2),k] != bar_refe or k == (cols-1)):
                bar_number += 1
                bar_number_plot.append(bar_number)
                bar_dist.append(dist_refe)                
                dist_refe = 0
            bar_refe = image[int(row/2),k]

        print(f"The number of bars is: {bar_number}")
        print(f"The average distance between the bars is: {sum(bar_dist)/len(bar_dist)}")
        cplt.create_plots(bar_number_plot,bar_dist,"BAR'S DISTANCES",'bar','pixels')


def main():
    coins = CoinSegmentation()

    coins.count_bars()



if __name__=='__main__':
    sys.exit(main())
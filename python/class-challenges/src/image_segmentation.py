# Built-int imports
import sys

# External imports 
import cv2 as cv
import numpy as np


class ImageSegmentation:
    def __init__(self, image):
        self.__image = image
        self.__rows, self.__cols = image.shape[:2]

    def nothing(self,x):
        pass

    def create_trackbar(self,name,title):
        return cv.createTrackbar(
            name,
            title,
            0,
            255,
            self.nothing
        )

    def binarization(self):
        cv.namedWindow("Image Binarization")
        self.create_trackbar("Gray-u","Image Binarization")
        self.create_trackbar("Gray-l","Image Binarization")

        image = cv.cvtColor(self.__image, cv.COLOR_RGB2GRAY)

        while(True):
            var_gray_upper = cv.getTrackbarPos("Gray-u","Image Binarization")
            var_gray_lower = cv.getTrackbarPos("Gray-l","Image Binarization")
    
            for i in iter(range(self.__rows)):
                for j in iter(range(self.__cols)):
                    if (var_gray_upper > image[i,j] and var_gray_lower < image[i,j]):
                        image[i,j] = 255
                    else:
                        image[i,j] = 0
                        
            cv.imshow("Image Binarization", image)
            image = cv.cvtColor(self.__image, cv.COLOR_RGB2GRAY)

            if (cv.waitKey(1) & 0xFF == ord('q')):
                break
        cv.destroyAllWindows()

    def color_segmentation(self):
        cv.namedWindow("Image Segmentation")
        self.create_trackbar("Blue-u", "Image Segmentation")
        self.create_trackbar("Blue-l","Image Segmentation")
        self.create_trackbar("Green-u","Image Segmentation")
        self.create_trackbar("Green-l","Image Segmentation")
        self.create_trackbar("Red-u","Image Segmentation")
        self.create_trackbar("Red-l","Image Segmentation")

        image = self.__image.copy()

        while(True):
            var_blue_upper = cv.getTrackbarPos("Blue-u", "Image Segmentation")
            var_blue_lower = cv.getTrackbarPos("Blue-l", "Image Segmentation")
            var_green_upper = cv.getTrackbarPos("Green-u", "Image Segmentation")
            var_green_lower = cv.getTrackbarPos("Green-l", "Image Segmentation")
            var_red_upper = cv.getTrackbarPos("Red-u", "Image Segmentation")
            var_red_lower = cv.getTrackbarPos("Red-l", "Image Segmentation")

            for i in iter(range(self.__rows)):
                for j in iter(range(self.__cols)):
                    if (var_blue_upper > image[i,j,0] and var_blue_lower < image[i,j,0] and \
                        var_green_upper > image[i,j,1] and var_green_lower < image[i,j,1] and \
                        var_red_upper > image[i,j,2] and var_red_lower < image[i,j,2]):
                        image[i,j] = 255
                    else:
                        image[i,j] = 0
                        
            cv.imshow("Image Segmentation", image)
            image = self.__image.copy()

            if (cv.waitKey(1) & 0xFF == ord('q')):
                break
        cv.destroyAllWindows()


def main():
    pass


if __name__=='__main__':
    sys.exit(main())
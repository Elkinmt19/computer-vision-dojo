# Built-int imports
import sys

# External imports 
import cv2 as cv
import numpy as np


class ImageSegmentation:
    """
    This is a python class that was implemented to make image binarization 
    and also to make image color segmentation in an easy way, this class 
    has a method called binarization which takes a gray image and maps the 
    best threshold to binarize it, and also has a method called color 
    segmentation which does the same as binarization but using a BGR image or 
    HSV image.
    :param image: BGR-image to binarized or to make a color segmentation.
    """
    def __init__(self, image):
        self.__image = image
        self.coordenates = np.array([[0,0],[0,0]])
        self.print_flag = False

    def nothing(self,x):
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

    def binarization(self):
        """"
        binarization method, this method takes an BGR image and finds the 
        best threshold values in order to binarize it.
        """
        cv.namedWindow("Image Binarization")
        self.create_trackbar("Gray-u","Image Binarization")
        self.create_trackbar("Gray-l","Image Binarization")

        image = cv.cvtColor(self.__image, cv.COLOR_BGR2GRAY)

        while(True):
            var_gray_upper = cv.getTrackbarPos("Gray-u","Image Binarization")
            var_gray_lower = cv.getTrackbarPos("Gray-l","Image Binarization")

            bin_image = cv.inRange(image, var_gray_lower, var_gray_upper)            
            cv.imshow("Image Binarization", bin_image)

            if (cv.waitKey(1) & 0xFF == ord('q')):
                break
        cv.destroyAllWindows()

    def color_segmentation(self, image_format = "BGR"):
        """
        color_segmentation method, this method takes a BGR image or 
        HSV image and finds the best threshold ni order to segmentate 
        an object in the image.
        """
        reset_image = False
        if (image_format == "HSV"):
            track_variables = ["Hue", "Saturation", "Value"]
            image = cv.cvtColor(self.__image, cv.COLOR_BGR2HSV)
            reset_image = True
        elif (image_format == "BGR"):
            track_variables = ["Blue", "Green", "Red"]
            image = self.__image.copy()

        cv.namedWindow("Image Segmentation")
        for i in iter(range(3)):
            self.create_trackbar(f"{track_variables[i]}-u", "Image Segmentation")
            self.create_trackbar(f"{track_variables[i]}-l", "Image Segmentation")

        format_variables = dict(zip(track_variables,[[0,0],[0,0],[0,0]]))

        while(True):
            for j in iter(range(3)):
                format_variables[track_variables[j]][0] = cv.getTrackbarPos(
                    f"{track_variables[j]}-u",
                    "Image Segmentation"
                )
                format_variables[track_variables[j]][1] = cv.getTrackbarPos(
                    f"{track_variables[j]}-l",
                    "Image Segmentation"
                )

            var_upper = np.array([format_variables[k][0] for k in format_variables.keys()])
            var_lower = np.array([format_variables[k][1] for k in format_variables.keys()])

            bin_image = cv.inRange(image, var_lower, var_upper)            
            cv.imshow("Image Segmentation", bin_image)

            if (cv.waitKey(1) & 0xFF == ord('q')):
                break
        cv.destroyAllWindows()

    def __click_mouse_callback(self, event, y, x, flags, param):
        """
        Click-mouse callback function to use a click event
        """
        if (event == cv.EVENT_LBUTTONDOWN):
            self.coordenates[0,0] = x
            self.coordenates[0,1] = y

        if (event == cv.EVENT_LBUTTONUP):
            self.coordenates[1,0] = x
            self.coordenates[1,1] = y
            self.print_flag = True

    def get_region_of_interest(self):
        """
        This a simple method which allows to get a specific region of
        interest from an image.
        """
        cv.namedWindow("Source Image")
        cv.setMouseCallback("Source Image", self.__click_mouse_callback)

        print("Getting the region of interest...")
        while (True):
            cv.imshow("Source Image", self.__image)
            if self.print_flag:
                print(f"Location: ({self.coordenates})")

                #! Fix the problem with negative-coordinates
                for j in iter(range(self.coordenates.shape[1])):
                    self.coordenates[:,j] = np.sort(self.coordenates[:,j], 0)

                self.roiImage = self.__image[
                    self.coordenates[0,0]:self.coordenates[1,0],
                    self.coordenates[0,1]:self.coordenates[1,1]
                ]

                cv.imshow("Source Image - ROI",self.roiImage)
                self.print_flag = False
            if (cv.waitKey(1) & 0xFF == ord('q')):
                break


def main():
    pass


if __name__=='__main__':
    sys.exit(main())
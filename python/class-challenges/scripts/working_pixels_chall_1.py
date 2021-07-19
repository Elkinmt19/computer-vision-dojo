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


class DolphinPlayingWithPixels:
    """
    Python class that has all the necessary methods to solve the first challenge
    of the EIA University's class of computer vision, the chanllenges was just some
    exercises to play with the pixels of an image.
    """
    def __init__(self):
        self.__delfin_path = self.get_image_path()
        self.image = cv.imread(self.__delfin_path, 1)
        self.__rows, self.__cols, self.__channels = self.image.shape


    def get_image_path(self):
        image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", "delfin.jpg")
        return image_relative_path

    def dolphin_separated(self, channel):
        new_image = cv.imread(self.__delfin_path, 1)
        for (i,j,c), _ in np.ndenumerate(new_image):
            if c == channel:
                new_image[i,j,c] = 255
        cv.imshow(f"New dolphin channel: {channel}", new_image)
    
    def dolphin_channels(self):
        """
        Method to show an image in its tree channels (Red, Green, Blue).
        """
        cv.imshow("Original dolphin", self.image)
        for i in iter(range(3)):
            self.dolphin_separated(i)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def dolphin_y_reverse(self):
        """
        Method for creating a mirror image on the y-axis.
        """
        new_image = cv.imread(self.__delfin_path, 1)

        for i in iter(range(self.__rows)):
            for c in iter(range(self.__channels)):
                for j in iter(range(self.__cols)):
                    new_image[i,j,c] = self.image[i,(self.__cols-1)-j, c]
        cv.imshow("Original dolphin", self.image)
        cv.imshow("y reverse dolphin", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def dolphin_x_reverse(self):
        """
        Method for creating a mirror image on the x-axis.
        """
        new_image = cv.imread(self.__delfin_path, 1)

        for j in iter(range(self.__cols)):
            for c in iter(range(self.__channels)):
                for i in iter(range(self.__rows)):
                    new_image[i,j,c] = self.image[(self.__rows-1)-i, j, c]
        cv.imshow("Original dolphin", self.image)
        cv.imshow("x reverse dolphin", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def dolphin_colorfull(self):
        """
        Method to created an image in a RBG mosaic with 4 squares.
        """
        new_image = cv.imread(self.__delfin_path, 1)

        for (i,j,c), _ in np.ndenumerate(new_image):
            if (i < self.__rows/2 and j >= self.__cols/2 and c == 0): 
                new_image[i,j,c] = 255
            if (i >= self.__rows/2 and j < self.__cols/2 and c == 1): 
                new_image[i,j,c] = 255
            if (i >= self.__rows/2 and j >= self.__cols/2 and c == 2): 
                new_image[i,j,c] = 255
        cv.imshow("Original dolphin", self.image)
        cv.imshow("Colorfull dolphin", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def dolphin_colorfull_by_user(self, px, py):
        """
        Method to create an image in an RGB mosaic, considering the width and the 
        height values entered by the user.

        :param px: height of the squares of the mosaic
        :para, py: weidth of the squares of the mosaic
        """
        new_image = cv.imread(self.__delfin_path, 1)
        print(f"px: {px} and py: {py}")
        x_count , y_count, c_count = (0,0,0)
        x_refe, y_refe, c_refe = (0,0,0)
        x_number = int(self.__rows/px)
        y_number = int(self.__cols/py)
        print(f"x_number: {x_number} y_number: {y_number}")

        def validated_limits(count, limit):
            if (count == limit):
                return 0
            return (count + 1)

        for i in iter(range(self.__rows)):
            x_count = i - x_refe
            for j in iter(range(self.__cols)):
                y_count = j - y_refe
                if (x_count <= px and y_count <= py):
                    new_image[i,j,c_count] = 255
                if (j == self.__cols - 1):
                    y_refe, c_count = (0,c_refe)
                if (y_count > py):
                    y_refe = j
                    c_count = validated_limits(c_count,2)
            if (x_count > px):
                c_refe = validated_limits(c_refe,2)
                x_refe = i        

        cv.imshow("Original dolphin", self.image)
        cv.imshow("Colorfull by user dolphin", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()




def main():
    """COMPUTER VISION - EIA UNIVERSITY
    First challenge of the EIA University's computer vision class.
    Run this scripts in order to see Elkin Guerra's solucion 
    of this test. 
    """
    epilog = """
    Related examples:
    More to come...
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__,
                                     epilog=epilog)
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-s', '--stage', dest='stage', required=True, choices=[
            "one","two","tree","four","five"
        ],
        help='The stage of the challenge you want to execute'
    )

    parser.add_argument(
        '-d', '--dimensions', dest='dimensions', required=False, nargs='+',
        help='Height and Weidth for the squares of the mosaic'
    )


    args = parser.parse_args()

    if args.dimensions != None:
        dimen = [int(x) for x in args.dimensions]
        
    print("Initializing program... ")
    dolphin = DolphinPlayingWithPixels()

    try:
        act = args.stage
        if act == "one":
            dolphin.dolphin_channels()
        elif act == "two":
            dolphin.dolphin_y_reverse()
        elif act == "tree":
            dolphin.dolphin_x_reverse()
        elif act == "four":
            dolphin.dolphin_colorfull()
        elif act == "five":
            dolphin.dolphin_colorfull_by_user(dimen[0],dimen[1])
            
    except:
        print("ERROR JUST HAPPEND")

    return 0


if __name__=='__main__':
    sys.exit(main())
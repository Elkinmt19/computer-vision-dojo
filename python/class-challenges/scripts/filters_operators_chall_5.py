# Built-int imports 
import os
import sys
import argparse

# External imports
import cv2 as cv
import numpy as np

# My own imports 
import image_segmentation as imgs
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

class FiltersOperators:
    """
    This is a python class that contains methods that allows to 
    separate a group of coins and also couns them with a very good 
    accuracy, also has a methd that filters a group of noisy images
    depending of some specifications.
    """
    def __init__(self):
        self.images = list()
        self.get_images()

    def get_image_path(self, image_name):
        image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", image_name)
        return image_relative_path

    def get_images(self):
        self.images.append(cv.imread(self.get_image_path("group_coins.png")))
        self.images.append(cv.imread(self.get_image_path("dots.png")))
        self.images.append(cv.imread(self.get_image_path("key.png")))
        self.images.append(cv.imread(self.get_image_path("walker.png")))

    def coins_binarization(self):
        img_seg = imgs.ImageSegmentation(self.images[0]).binarization()

    def separate_coins(self):
        """
        This is a simple python method that separates a group of coins appling 
        some image filters and then getting the number of coins using contours.
        """
        gray_image = cv.cvtColor(self.images[0], cv.COLOR_BGR2GRAY)
        _,binary_image = cv.threshold(gray_image, 133, 255, cv.THRESH_BINARY_INV)
        open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50,50))
        erode_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10,14))
        dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))

        # erode filter - to erode an image
        erode_image = cv.erode(binary_image, erode_kernel, iterations=1)

        # Others import filters to take into account
        img_open = cv.morphologyEx(erode_image,cv.MORPH_OPEN,open_kernel, iterations=1)

        # Dilate filter - to dilate an image
        dilate_image = cv.dilate(img_open, dilate_kernel, iterations=1)

        # Find the contours of the image
        contours, hier = cv.findContours(
            dilate_image.copy(),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE
        )

        print(f"The number of coins is {len(contours)}")

        # Draw the comtours in the image
        contour_image = np.zeros((dilate_image.shape[0], dilate_image.shape[1], 3), np.uint8)

        for cnt in contours:
            area = cv.contourArea(cnt)
            if (area > 0):
                cv.drawContours(contour_image, cnt, -1, (255,0,0), 2)

        # Show all the images
        cv.imshow("Original image", self.images[0])
        cv.imshow("Filtered image (Coins separated)", dilate_image)
        cv.imshow("Count of the coins", contour_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def filter_noisy_images(self):
        """
        This is a simple python method that allows to modify a group of 
        noisy images using some image filters.
        """
        binary_image = list()
        # Filter dots 
        for i in iter(range(1,4)):
            gray_image = cv.cvtColor(self.images[i], cv.COLOR_BGR2GRAY)
            _,binary_image_buff = cv.threshold(gray_image, 133, 255, cv.THRESH_BINARY)
            binary_image.append(binary_image_buff)

        def filter_dots(img):
            erode_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (32,32))
            dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30,30))

            # erode filter - to erode an image
            erode_image = cv.erode(img, erode_kernel, iterations=1)

            # dilate filter - to dilate an image
            dilate_image = cv.dilate(erode_image, dilate_kernel, iterations=1)

            cv.imshow("Original dots image", img)
            cv.imshow("Filtered dots image", dilate_image)
            cv.waitKey(0)

        def filter_key(img):
            kernel = np.ones((4,4), np.uint8)
            erode_kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))

            # erode filter - to erode an image
            erode_image = cv.erode(img, erode_kernel, iterations=1)

            img_gradient = cv.morphologyEx(erode_image, cv.MORPH_GRADIENT, kernel, iterations=1)

            cv.imshow("Original dots image", img)
            cv.imshow("Filtered dots image", img_gradient)
            cv.waitKey(0)

        def filter_walker(img):
            erode_kernel = cv.getStructuringElement(cv.MORPH_RECT , (6,6))
            dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))

            # erode filter - to erode an image
            erode_image = cv.erode(img, erode_kernel, iterations=1)

            # dilate filter - to dilate an image
            dilate_image = cv.dilate(erode_image, dilate_kernel, iterations=1)

            cv.imshow("Original dots image", img)
            cv.imshow("Filtered dots image", dilate_image)
            cv.waitKey(0)
        
        filter_dots(binary_image[0])
        filter_key(binary_image[1])
        filter_walker(binary_image[2])
        cv.destroyAllWindows()




def main():
    """COMPUTER VISION - EIA UNIVERSITY
    Fifth challenge of the EIA University's computer vision class.
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
        epilog=epilog
    )
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-s', '--stage', dest='stage', required=True, choices=[
            "one","two"
        ],
        help='The stage of the challenge you want to execute'
    )

    args = parser.parse_args()

    print("Initializing program... ")
    filter_operator = FiltersOperators()

    try:
        act = args.stage
        if act == "one":
            filter_operator.separate_coins()
        elif act == "two":
            filter_operator.filter_noisy_images()
            
    except:
        print("ERROR JUST HAPPEND")

    return 0

if __name__=='__main__':
    sys.exit(main())

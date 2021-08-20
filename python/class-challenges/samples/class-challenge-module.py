#Built-int imports
import sys
import os

# External imports
import cv2 as cv

# My own imports 
import image_segmentation as igs
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()


image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", "pills_2.png")
img = cv.imread(image_relative_path,1)

# cv.imshow("test gray image", cv.cvtColor(img,cv.COLOR_RGB2GRAY))

# seg = igs.ImageSegmentation(img).binarization()
seg = igs.ImageSegmentation(img).color_segmentation("HSV")
#Built-int imports
import sys
import os

# External imports
import cv2 as cv

# My own imports 
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()


image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", "Tony_Stark.jpeg")
img = cv.imread(image_relative_path)
if img is None:
    sys.exit("Could not read the image.")
cv.imshow("Display window", img)
k = cv.waitKey(0)
if k == ord("s"):
    pass
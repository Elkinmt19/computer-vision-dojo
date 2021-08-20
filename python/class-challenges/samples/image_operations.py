# Built-int imports
import os

# External imports 
import cv2 as cv
import numpy as np

# My Own imports
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

def get_path_image(image_name):
    return os.path.join(
            ASSETS_FOLDER, "imgs", image_name)

image_1 = cv.imread(get_path_image("pills_1.png"),1)
image_2 = cv.imread(get_path_image("pills_2.png"),1)


print(f"Size of the pills 1: {image_1.shape[:2]}")
print(f"Size of the pills 2: {image_2.shape[:2]}")

# Making the image operations using numpy operators
img_sum = image_1 + image_2
img_diff = image_1 - image_2
img_mult = image_1 * image_2

# Making the image operations using Open CV operators
img_sum_cv = cv.add(image_1, image_2)
img_diff_cv = cv.subtract(image_1, image_2)
img_mult_cv = cv.multiply(image_1, image_2)


cv.imshow("Images Sum", img_sum)
cv.imshow("Images Sum CV", img_sum_cv)
cv.imshow("Images Diff", img_diff)
cv.imshow("Images Diff CV", img_diff_cv)
cv.imshow("Images Mult", img_mult)
cv.imshow("Images Mult CV", img_mult_cv)

cv.waitKey()
# Built-int imports
import os

# External imports 
import cv2 as cv

# My Own imports
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

def get_path_image(image_name):
    return os.path.join(
            ASSETS_FOLDER, "imgs", image_name)

image_1 = cv.imread(get_path_image("frame_1.png"),1)
image_2 = cv.imread(get_path_image("frame_30.png"),1)


print(f"Size of the pills 1: {image_1.shape[:2]}")
print(f"Size of the pills 2: {image_2.shape[:2]}")


img_sum = image_1 + image_2
img_diff = image_1 - image_2
img_mult = image_1 * image_2

cv.imshow("Images Sum", img_sum)
cv.imshow("Images Diff", img_diff)
cv.imshow("Images Mult", img_mult)

cv.waitKey()
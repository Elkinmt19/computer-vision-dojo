# Built-int imports
import os

# External imports 
import cv2 as cv
import numpy as np

# My Own imports
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", "eGaIy.jpg")

img = cv.imread(image_relative_path, 0)
img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]  # ensure binary
print(img.shape)
num_labels, labels_im = cv.connectedComponents(img)

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    print(label_hue.shape)
    blank_ch = 255*np.ones_like(label_hue)
    print(blank_ch.shape)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv.imshow('labeled.png', labeled_img)
    cv.waitKey()

imshow_components(labels_im)

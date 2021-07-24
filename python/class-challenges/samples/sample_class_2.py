# Built-int imports
import os

# External imports 
import cv2 as cv
import numpy as np

# My Own imports
import sample_class_1 as sc1
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

"""TOPICS (7-24-2021):
* How to use the click-mouse event in order to select a specific area of an image
* How to use and create a track-bar event in order to get a value of a specific channel of an image
"""
coordenates = np.array([[0,0],[0,0]])
print_flag = False
def click_mouse_callback(event, y, x, flags, param):
    """
    Click-mouse callback function to use a click event
    """
    global coordenates
    global print_flag
    if (event == cv.EVENT_LBUTTONDOWN):
        coordenates[0,0] = x
        coordenates[0,1] = y

    if (event == cv.EVENT_LBUTTONUP):
        coordenates[1,0] = x
        coordenates[1,1] = y
        print_flag = True

# Function that is necessary for the trackbar event
def nothing(x):
    """
    nothing function that is necessary for the trackbar event
    """
    pass

def click_mouse():
    global coordenates
    global print_flag
    # In order to use the click-mouse event, first it's necessary to create a windows like is shown below
    cv.namedWindow("Tony Stark")
    # After created the windows it's important to set the callback
    cv.setMouseCallback("Tony Stark", click_mouse_callback)

    while (True):
        image_relative_path = os.path.join(
            ASSETS_FOLDER, "imgs", "Tony_Stark.jpeg")
        colorImage = sc1.load_image(image_relative_path, 1)
        resizeImage = cv.resize(colorImage, (640,480))
        sc1.show_image("Tony Stark", resizeImage)
        if print_flag:
            print(f"Location: ({coordenates})")

            #! Fix the problem with negative-coordinates
            for j in iter(range(coordenates.shape[1])):
                coordenates[:,j] = np.sort(coordenates[:,j], 0)

            roiImage = resizeImage[
                coordenates[0,0]:coordenates[1,0],
                coordenates[0,1]:coordenates[1,1]
            ]
            cv.imshow("Tony ROI",roiImage)
            print_flag = False
        if (cv.waitKey(1) & 0xFF == ord('q')):
            break

def track_bar():
    # In order to use the trackbar event, first it's necessary to create a windows like is shown below
    cv.namedWindow("Tony Stark")
    # After created the windows it's important to set the callback
    cv.createTrackbar("var_1", "Tony Stark", 0, 255, nothing)
    cv.createTrackbar("var_2", "Tony Stark", 0, 255, nothing)

    image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", "Tony_Stark.jpeg")
    colorImage = sc1.load_image(image_relative_path, 1)
    resizeImage = cv.resize(colorImage, (640,480))
    sc1.show_image("Tony Stark", resizeImage)

    copyImage = resizeImage.copy()
    h,w = resizeImage.shape[:2]

    while (True):
        val_track_1 = cv.getTrackbarPos("var_1","Tony Stark")
        val_track_2 = cv.getTrackbarPos("var_2","Tony Stark")
        for i in iter(range(h)):
            for j in iter(range(w)):
                val_pixel = resizeImage[i,j]
                if (val_track_1 > val_pixel[0] and val_track_2 < val_pixel[0]):
                    copyImage[i,j] = 255
                else:
                    copyImage[i,j] = 0
        sc1.show_image("Tony Copy", copyImage)
        copyImage = resizeImage.copy()
        if (cv.waitKey(1) & 0xFF == ord('q')):
            break

def main():
    # Implementation of the 'click_mouse()' function
    click_mouse()

    # Implementation of the 'track_bar()' function 
    track_bar() 

if __name__ == '__main__':
    main()
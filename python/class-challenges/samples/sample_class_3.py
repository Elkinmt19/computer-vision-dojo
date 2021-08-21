# Built-int imports
import os

# External imports 
import cv2 as cv
import numpy as np

# My Own imports
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

def nothing(x):
        pass

def create_trackbar(name,title):
    return cv.createTrackbar(
        name,
        title,
        0,
        255,
        nothing
    )


def main():
    cv.namedWindow("Fachada HSV - before")
    create_trackbar("h-u", "Fachada HSV - before")
    create_trackbar("h-l","Fachada HSV - before")
    create_trackbar("s-u","Fachada HSV - before")
    create_trackbar("s-l","Fachada HSV - before")
    create_trackbar("v-u","Fachada HSV - before")
    create_trackbar("v-l","Fachada HSV - before")


    image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", "fachada1.png")
    colorImage = cv.imread(image_relative_path, 1)

    while True:
        hsv_image = cv.cvtColor(colorImage, cv.COLOR_BGR2HSV)

        cv.imshow("Fachada RGB - before", colorImage)
        cv.imshow("Fachada HSV - before", hsv_image)

        var_h_upper = cv.getTrackbarPos("h-u", "Fachada HSV - before")
        var_h_lower = cv.getTrackbarPos("h-l", "Fachada HSV - before")
        var_s_upper = cv.getTrackbarPos("s-u", "Fachada HSV - before")
        var_s_lower = cv.getTrackbarPos("s-l", "Fachada HSV - before")
        var_v_upper = cv.getTrackbarPos("v-u", "Fachada HSV - before")
        var_v_lower = cv.getTrackbarPos("v-l", "Fachada HSV - before")

        lower = np.array([var_h_lower,var_s_lower,var_v_lower])
        upper = np.array([var_h_upper,var_s_upper,var_v_upper])

        # Faster binarization method (binary mask)
        bin_image = cv.inRange(hsv_image, lower, upper) # What an awesome function 
        cv.imshow("Binarization image", bin_image)

        # cv.imshow("Fachada HSV - after", hsv_image)
        # bgr_image = cv.cvtColor(colorImage, cv.COLOR_HSV2BGR)
        # cv.imshow("Fachada RGB - after", bgr_image)

        if (cv.waitKey(1) & 0xFF == ord('q')):
            break
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
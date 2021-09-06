# Built-in imports 
import os
import sys
import math
import argparse

# External imports
import cv2 as cv
import numpy as np

# My own imports 
import image_segmentation as imgs
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

class PoolGame:
    def __init__(self):
        self.video = cv.VideoCapture(self.get_video_path("pool.mp4"))
        self.get_images()

    def get_video_path(self, video_name):
        video_relative_path = os.path.join(
        ASSETS_FOLDER, "videos", video_name)
        return video_relative_path

    def get_image_path(self, image_name):
        image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", image_name)
        return image_relative_path

    def get_images(self):
        self.images = list()
        self.images.append(cv.imread(self.get_image_path("holes.png"),1))
        self.images.append(cv.imread(self.get_image_path("balls.png"),1))

    def watch_video(self):
        while(self.video.isOpened()):
            ret, frame = self.video.read()
            if (ret == False):
                print("There is no image, the video is stoped")
                break
            cv.imshow("Frame", frame)
            if (cv.waitKey(50) & 0xFF == ord('q')):
                break

        self.video.release()
        cv.destroyAllWindows()

    def get_umbral_values_holes_segmentation(self):
        img_seg = imgs.ImageSegmentation(self.images[0]).binarization()

    def holes_segmentation(self):
        gray_image = cv.cvtColor(self.images[0], cv.COLOR_BGR2GRAY)
        _,binary_image = cv.threshold(gray_image, 27, 255, cv.THRESH_BINARY_INV)

        erode_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE , (20,20))
        dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20,20))

        # erode filter - to erode an image
        erode_image = cv.erode(binary_image, erode_kernel, iterations=1)

        # dilate filter - to dilate an image
        dilate_image = cv.dilate(erode_image, dilate_kernel, iterations=1)

        # Find the contours of the image
        contours, hier = cv.findContours(
            dilate_image.copy(),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE
        )

        # Draw the comtours in the image
        contour_image = np.zeros((dilate_image.shape[0], dilate_image.shape[1], 3), np.uint8)

        for cnt in contours:
            area = cv.contourArea(cnt)
            ratio = int(math.sqrt(area/math.pi))
            if (area > 0):
                M = cv.moments(cnt)
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                cv.circle(contour_image, (cx,cy), ratio, (0,0,255), 2)
                cv.circle(contour_image, (cx,cy), 1, (0,255,0), 2)

        cv.imshow("Binary Image", contour_image)
        cv.imshow("Original Image", dilate_image)

        cv.waitKey(0)
        cv.destroyAllWindows()

    def get_umbral_values_balls_segmentation(self):
        img_seg = imgs.ImageSegmentation(self.images[1]).binarization()



def main():
    pool_game = PoolGame().get_umbral_values_balls_segmentation()

if __name__ == "__main__":
    sys.exit(main())
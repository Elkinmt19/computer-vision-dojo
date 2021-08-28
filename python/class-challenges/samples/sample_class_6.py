# Built-in imports 
import os 
import glob 
import time
import sys

# External imports
import cv2 as cv

# My Own imports
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

def load_multiple_images():
    # Universal path depending of the extention of the image
    image_relative_path = os.path.join(
            ASSETS_FOLDER, "imgs", "*.png")

    for img in glob.glob(image_relative_path):
        color_img = cv.imread(img,1)
        cv.imshow("Color Image", color_img)
        cv.waitKey(0)
    cv.destroyAllWindows()

def working_with_videos():
    video_relative_path = os.path.join(
            ASSETS_FOLDER, "videos", "biomedica.mp4")

    capture = cv.VideoCapture(video_relative_path)
    time.sleep(2)

    while(capture.isOpened()):
        ret, frame = capture.read()
        if (ret == False):
            print("There is no image, the video is stop")
            break
        cv.imshow("Frame", frame)
        cv.waitKey(1)

    capture.release()
    cv.destroyAllWindows()

def working_with_web_camera():
    capture = cv.VideoCapture(0)
    time.sleep(2)

    while(capture.isOpened()):
        ret, frame = capture.read()
        if (ret == False):
            print("There is no image, the video is stop")
            break
        cv.imshow("Frame", frame)
        cv.waitKey(1)

    capture.release()
    cv.destroyAllWindows()

def main():
    working_with_web_camera()

if __name__ == "__main__":
    sys.exit(main())
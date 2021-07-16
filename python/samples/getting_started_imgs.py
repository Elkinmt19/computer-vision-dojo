#Built-int imports
import sys

# Others imports
import cv2 as cv

img = cv.imread("python/assets/imgs/Tony_Stark.jpeg")
if img is None:
    sys.exit("Could not read the image.")
cv.imshow("Display window", img)
k = cv.waitKey(0)
if k == ord("s"):
    pass
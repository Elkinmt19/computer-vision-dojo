# Built-in imports 
import os

# External imports
import numpy as np
import cv2 as cv

# My Own imports
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", "Tony_Stark.jpeg")
colorImage = cv.imread(image_relative_path, 1)
img = cv.resize(colorImage, (512,512))

# Draw a diagonal blue line with thickness of 5 px
cv.line(img,(0,0),(511,511),(255,0,0),5)

# Draw a rectangle in green of 3 px
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)

# Draw a filled circle in red 
cv.circle(img,(447,63), 63, (0,0,255), -1)

# Draw a filled ellipse 
cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

# Draw a polylines
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(0,255,255))

# Put text in the image
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'Tony Stark',(80,500), font, 2,(255,255,255),2,cv.LINE_AA)

cv.imshow("Resulted Image", img)
cv.waitKey(0)
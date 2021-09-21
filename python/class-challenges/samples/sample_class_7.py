# Built-in imports 
import os
from glob import glob

# External imports 
import numpy as np
import cv2 as cv
import xlsxwriter

# My Own imports
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()


# Define the first cell of the excel file 
row = 0
col = 0

# Counter that allows to move between the excel's columns
i=1

# Create a new excel file with a new sheet to work with
path_dataset_file = os.path.join(
    ASSETS_FOLDER, "xlsx", "Numbers_dataset.xlsx")
workbook = xlsxwriter.Workbook(path_dataset_file)
worksheet = workbook.add_worksheet('Num_0')


# Universal path depending of the extention of the image
path_zero_number = os.path.join(
    ASSETS_FOLDER, "imgs", "num/0/0 (*.png")

# Function to find the images of the dataset
img_names = glob(path_zero_number)

for fn in img_names:
    img = cv.imread(fn, 1)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray,5)
    _,binary_image = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    drawing = np.zeros(img.shape,np.uint8) # Image to draw the contours

    # Get the contours of the images
    contours,hierarchy = cv.findContours(binary_image.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    # Get the 7 moments of hu and other characteristics
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        roi_image = binary_image[
            y:y+h,
            x:x+w
        ]

        # Get the region of interest of each one of the images 
        img_resize = cv.resize(roi_image,(30, 60), interpolation = cv.INTER_CUBIC)
        _,binary_image_res = cv.threshold(img_resize,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
        cv.imshow("img_resize", binary_image_res)
        cv.waitKey(1)

        # Get the contours again in order to make sure that the object has not been corructed
        contours_2,hierarchy_2 = cv.findContours(binary_image_res.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        for cnt_2 in contours_2:
            x,y,w,h = cv.boundingRect(cnt_2)
            M = cv.moments(cnt_2)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            A = cv.contourArea(cnt_2)
            p = cv.arcLength(cnt_2,True)
            Comp = A/float(p*p)
            RA = w/float(h)
            Hu = cv.HuMoments(M)

            VectorCarac = np.array([A, p, Comp, RA, Hu[0][0], Hu[1][0], Hu[2][0], Hu[3][0], Hu[4][0], Hu[5][0], Hu[6][0]], dtype = np.float32)
        
            for carac in (VectorCarac):
                worksheet.write(row, col, "A")
                worksheet.write(row, i, carac)
                i=i+1
            i=1
            row += 1

# Close the excel file
workbook.close()
cv.destroyAllWindows()
import csv
import cv2
import glob
import numpy as np
import os.path
import shutil

path = 'imagesYOLO/'

DoData = True

path_2 = 'images_Data_Augmentation/'
COLOR = [(255, 0, 0), (0, 0, 255)]
EXIT = False
jj = 0


def writeCSV(fileTXT, data):
    fileTXT = fileTXT[:fileTXT.find('.')] + '.txt'
    #print(fileTXT)
    myFile = open(fileTXT, 'a', newline = '')
    with myFile:
        writer = csv.writer(myFile, delimiter=' ')
        writer.writerow(data)
    #print("Writing complete")

def filtrar_imagen_GaussBlur(Img):
    return (cv2.GaussianBlur(Img,(13,13),13,13))
     

def filtrar_imagen_MedianBlur(Img):
    return ( cv2.medianBlur(Img,13))
     

def equalize_image_YuV(img):
    img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    return (cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR))

def convertir_imagen_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hsv[:,:,1] = cv2.equalizeHist(hsv[:,:,1])
    return (cv2.cvtColor(hsv, cv2.COLOR_HLS2BGR))

def convertir_imagen_RGB(img):   
    return (cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def adjust_gamma(img, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(img, table)
    
        
def readCSV(fileTXT):
    global EXIT, jj
    with open(fileTXT) as File:        
        fileImg = fileTXT[:fileTXT.find('.')] + '.jpg'        
        img_data = cv2.imread(fileImg)
        #H, W = img.shape[:2]
        print(jj)
        ii = 0    
        if DoData:
            img_Gauss_Blur = filtrar_imagen_GaussBlur(img_data)                  
            img_Median_Blur = filtrar_imagen_MedianBlur(img_data)
            img_ecualize_YuV = equalize_image_YuV(img_data)
            img_ecualize_HSL = convertir_imagen_hsv(img_data)
            img_ecualize_RGB = convertir_imagen_RGB (img_data)
##            cv2.imshow("img_Gauss_Blur", cv2.resize(img_Gauss_Blur, (320, 240)))
##            cv2.imshow("img_Median_Blur", cv2.resize(img_Median_Blur, (320, 240)))
##            cv2.imshow("img_ecualize_YuV", cv2.resize(img_ecualize_YuV, (320, 240)))
##            cv2.imshow("img_ecualize_HSL", cv2.resize(img_ecualize_HSL, (320, 240)))
##            cv2.imshow("img_ecualize_RGB", cv2.resize(img_ecualize_RGB, (320, 240)))
##            cv2.imshow("image", cv2.resize(img_data, (320, 240)))                  

            fileTXT_2 = path_2 + "img_Gauss_Blur_" + str(jj) + '.txt'
            fileImg_2 = path_2 +"img_Gauss_Blur_" + str(jj) + '.jpg'
            shutil.copy(fileTXT, fileTXT_2)
            cv2.imwrite(fileImg_2,img_Gauss_Blur)

            fileTXT_2 = path_2 + "img_Median_Blur" + str(jj) + '.txt'
            fileImg_2 = path_2 +"img_Median_Blur" + str(jj) + '.jpg'
            shutil.copy(fileTXT, fileTXT_2)
            cv2.imwrite(fileImg_2,img_Median_Blur)

            fileTXT_2 = path_2 + "img_ecualize_YuV" + str(jj) + '.txt'
            fileImg_2 = path_2 +"img_ecualize_YuV" + str(jj) + '.jpg'
            shutil.copy(fileTXT, fileTXT_2)
            cv2.imwrite(fileImg_2,img_ecualize_YuV)

            fileTXT_2 = path_2 + "img_ecualize_HSL" + str(jj) + '.txt'
            fileImg_2 = path_2 +"img_ecualize_HSL" + str(jj) + '.jpg'
            shutil.copy(fileTXT, fileTXT_2)
            cv2.imwrite(fileImg_2,img_ecualize_HSL)

            fileTXT_2 = path_2 + "img_ecualize_RGB" + str(jj) + '.txt'
            fileImg_2 = path_2 +"img_ecualize_RGB" + str(jj) + '.jpg'
            shutil.copy(fileTXT, fileTXT_2)
            cv2.imwrite(fileImg_2,img_ecualize_RGB)
            
            # loop over various values of gamma
            for gamma in np.arange(1.25, 3.0, 0.25):
                
                # ignore when gamma is 1 (there will be no change to the image)
                if gamma == 1:
                        continue
         
                # apply gamma correction and show the images
                gamma = gamma if gamma > 0 else 0.1
                adjusted = adjust_gamma(img_data, gamma=gamma)
                fileTXT_2 = path_2 + "adjusted_gamma" + str(jj) + "_" + str(ii) + '.txt'
                fileImg_2 = path_2 +"adjusted_gamma" + str(jj) + "_" + str(ii) + '.jpg'
                shutil.copy(fileTXT, fileTXT_2)
                cv2.imwrite(fileImg_2,adjusted)
                #cv2.imshow("adjusted_gamma",cv2.resize(adjusted, (320, 240)))
                #cv2.waitKey(1)
                ii = ii + 1
            
            
                
##                writeCSV(fileTXT_2, row)                    
##                if not os.path.isfile(fileImg_2):
##                    writeCSV(fileList, [path_2 + fileImg])
##                    shutil.copy(fileImg, fileImg_2) 
        jj = jj + 1    
        #cv2.imshow("image", cv2.resize(img, (920, 640)))                  
        key = cv2.waitKey(1)      
        if key == ord('q'):
            EXIT = True
        #cv2.destroyAllWindows()

if not os.path.exists(path_2):
    os.mkdir(path_2)
    
for p in glob.glob(path + "\*.txt"):
    #print(p)
    #writeCSV(fileList, [p])
    readCSV(p)
    if EXIT:
        break
print("FIN")

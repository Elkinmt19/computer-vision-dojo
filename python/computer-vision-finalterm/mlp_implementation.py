# Built-in imports 
import sys
import os
import math
from matplotlib.pyplot import contour

# External imports
import numpy as np 
import cv2 as cv
import joblib

# My own imports 
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

class MlpImplementation:
    def __init__(self):
        self.load_models()
        self.load_image()

    def load_models(self):
        self.ss_model_numbers = joblib.load(self.models_path("model_scaling_all_numbers.pkl"))
        self.pca_model_numbers = joblib.load(self.models_path("model_pca_all_numbers.pkl"))
        self.mlp_model_numbers = joblib.load(self.models_path("model_mlp_numbers.pkl"))

        self.ss_model_letters = joblib.load(self.models_path("model_scaling_all_letters.pkl"))
        self.pca_model_letters = joblib.load(self.models_path("model_pca_all_letters.pkl"))
        self.mlp_model_letters = joblib.load(self.models_path("model_mlp_letters.pkl"))

    def load_image(self):
        image_path = os.path.join(
            ASSETS_FOLDER,
            "imgs/Test",
            "CC_6.jpg"
        )
        self.image = cv.imread(image_path, 0)

        _,self.image = cv.threshold(self.image,100,255,cv.THRESH_BINARY_INV)

    def models_path(self, model_name):
        model_path = os.path.join(
            ASSETS_FOLDER,
            "models",
            model_name
        )
        return model_path

    def get_con_letters(self):
        # Binarize the image
        binary_image = self.image

        binary_image = cv.resize(binary_image, None, fx=10, fy=10)

        num_image = binary_image[
            115:400,
            500:870
        ]

        self.num_image = num_image

        letters_image = binary_image[
            115:400,
            57:462
        ]

        self.letters_image = letters_image
        # Get the contours of the images
        self.numbers, _ = cv.findContours(num_image.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        self.letters, _ = cv.findContours(letters_image.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        # Filter the contours
        self.letters = [x for x in self.letters if (x.shape[0] >= 13)]
        self.numbers = [x for x in self.numbers if (x.shape[0] >= 20)]
        

    def corner_detection(self, image):
        gray = np.float32(image)
        dst = cv.cornerHarris(gray,2,3,0.04)
        #result is dilated for marking the corners, not important
        dst = cv.dilate(dst,None)
        # Threshold for an optimal value, it may vary depending on the image.
        key_variable = dst>0.01*dst.max()
        
        x = list(np.where(key_variable == True)[0])
        y = list(np.where(key_variable == True)[1])

        xmean = sum(x)
        ymean = sum(y)

        return xmean, ymean

    def extract_features(self, img):
        # Define kernels for the filters
        erode_kernel = cv.getStructuringElement(cv.MORPH_RECT , (3,3))
        dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))

    
        blur = cv.medianBlur(img,5)

        # erode filter - to erode an image
        erode_image = cv.erode(blur, erode_kernel, iterations=1)

        # dilate filter - to dilate an image
        dilate_image = cv.dilate(erode_image, dilate_kernel, iterations=1)

        _,binary_image = cv.threshold(dilate_image,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Get the contours of the images
        contours, _ = cv.findContours(binary_image.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        # Process the contornous
        contours = [x for x in contours if (x.shape[0] >= 20)]

        # Get the 7 moments of hu and other characteristics
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
            cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
            roi_image = binary_image[
                y:y+h,
                x:x+w
            ]

            # Get the region of interest of each one of the images 
            img_resize = cv.resize(roi_image,(30, 70), interpolation = cv.INTER_CUBIC)
            _,binary_image_res = cv.threshold(img_resize,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)

            # Calculate more features based on the corners founded by Harris algorithm
            corner_x, corner_y = self.corner_detection(binary_image_res.copy())

            # Get the contours again in order to make sure that the object has not been corructed
            contours_2, _ = cv.findContours(binary_image_res.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

            for cnt_2 in contours_2:
                try:
                    # Calculate the object's features
                    x,y,w,h = cv.boundingRect(cnt_2)
                    M = cv.moments(cnt_2)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    A = cv.contourArea(cnt_2)
                    p = cv.arcLength(cnt_2,True)
                    Comp = A/float(p*p)
                    Circ = p**2/(4*math.pi*A)
                    r = A/(w*h)
                    RA = w/float(h)
                    Hu = cv.HuMoments(M)

                    # # Calculate additional features
                    diago_mean = 0
                    diago_mean1 = 0
                    diago_mean2 = 0
                    diago_mean3 = 0
                    diago_mean4 = 0
                    y_mean = 0
                    x_mean = 0
                    y_mean_half_l = 0
                    x_mean_half_l = 0
                    y_mean_half_u = 0
                    x_mean_half_u = 0
                    y_mean_start = 0
                    x_mean_start = 0
                    y_mean_last = 0
                    x_mean_last = 0
                    for i in iter(range(binary_image_res.shape[0])):
                        for j in iter(range(binary_image_res.shape[1])):
                            if (i == j):
                                diago_mean += binary_image_res[i,j]
                                diago_mean1 += binary_image_res[i - 10,j]
                                diago_mean2 += binary_image_res[i + 10,j]
                                diago_mean3 += binary_image_res[i - 20,j]
                                diago_mean4 += binary_image_res[i + 20,j]
                            if (i == int(binary_image_res.shape[0]/2)):
                                y_mean += binary_image_res[i,j]
                            if (j == int(binary_image_res.shape[1]/2)):
                                x_mean += binary_image_res[i,j]
                            if (i == int(binary_image_res.shape[0]/4)):
                                y_mean_half_l += binary_image_res[i,j]
                            if (j == int(binary_image_res.shape[1]/4)):
                                x_mean_half_l += binary_image_res[i,j] 
                            if (i == int((binary_image_res.shape[0]*3)/4)):
                                y_mean_half_u += binary_image_res[i,j]
                            if (j == int((binary_image_res.shape[1]*3)/4)):
                                x_mean_half_u += binary_image_res[i,j]
                            if (i == 0):
                                y_mean_start += binary_image_res[i,j]
                            if (j == 0):
                                x_mean_start += binary_image_res[i,j] 
                            if (i == int(binary_image_res.shape[0]-1)):
                                y_mean_last += binary_image_res[i,j]
                            if (j == int(binary_image_res.shape[1]-1)):
                                x_mean_last += binary_image_res[i,j]  

                    # Vector with the object's features
                    VectorCarac = np.array([
                        A,
                        p,
                        Comp,
                        Circ,
                        r,
                        RA,
                        diago_mean,
                        diago_mean1,
                        diago_mean2,
                        diago_mean3,
                        diago_mean4,
                        corner_x,
                        corner_y,
                        y_mean,
                        x_mean,
                        y_mean_half_l,
                        x_mean_half_l,
                        y_mean_half_u,
                        x_mean_half_u,
                        y_mean_start,
                        x_mean_start,
                        y_mean_last,
                        x_mean_last,
                        Hu[0][0],
                        Hu[1][0],
                        Hu[2][0],
                        Hu[3][0],
                        Hu[4][0],
                        Hu[5][0],
                        Hu[6][0]],
                        dtype = np.float32
                    )
                    VectorCarac = VectorCarac.reshape(1,-1)
                    return VectorCarac
                except:
                    print("An error has occured")

    def mlp_implementation(self):
        NUMBERS_LIST = [
            "0","1",
            "2","3",
            "4","5",
            "6","7",
            "8","9"
        ]

        LETTERS_LIST = [
            "A","B",
            "C","D",
            "E","F",
            "G","H",
            "I","J",
            "K","L",
            "M","N",
            "O","P",
            "Q","R",
            "S","T",
            "U","V",
            "W","X",
            "Y","Z"
        ]

        self.get_con_letters()
        for lt in self.numbers:
            x,y,w,h = cv.boundingRect(lt)

            M = cv.moments(lt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            roiImage = self.num_image[
                y:y+h,
                x:x+w
            ]

            X = self.extract_features(roiImage)

            # Scaling the data
            X = self.ss_model_numbers.transform(X)

            # Make PCA analysis
            X = self.pca_model_numbers.transform(X)

            # Predicting the letter of the image
            result = self.mlp_model_numbers.predict(X)

            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(self.num_image, NUMBERS_LIST[int(result[0])],(cx,cy+50), font, 1,(255,255,0),2,cv.LINE_AA)
            print(NUMBERS_LIST[int(result[0])])

        for lt in self.letters:
            x,y,w,h = cv.boundingRect(lt)

            M = cv.moments(lt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            roiImage = self.letters_image[
                y:y+h,
                x:x+w
            ]

            X = self.extract_features(roiImage)

            # Scaling the data
            X = self.ss_model_letters.transform(X)

            # Make PCA analysis
            X = self.pca_model_letters.transform(X)

            # Predicting the letter of the image
            result = self.mlp_model_letters.predict(X)

            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(self.letters_image, LETTERS_LIST[int(result[0])],(cx,cy+50), font, 1,(255,255,0),2,cv.LINE_AA)
            print(LETTERS_LIST[int(result[0])])

        # Put text in the image
        cv.imshow("Predictive letters", self.letters_image)
        cv.imshow("Predictive numbers", self.num_image)
        cv.waitKey(0)



def main():
    mlp_implementation = MlpImplementation().mlp_implementation()

if __name__ == "__main__":
    sys.exit(main())


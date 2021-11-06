# Built-in imports 
import sys
import os
import math

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
        self.ss_model = joblib.load(self.models_path("model_scaling_all_characters.pkl"))
        self.pca_model = joblib.load(self.models_path("model_pca_all_characters.pkl"))
        self.mlp_model = joblib.load(self.models_path("model_mlp_characters.pkl"))

    def load_image(self):
        image_path = os.path.join(
            ASSETS_FOLDER,
            "imgs/tests/test_50",
            "test_50.png"
        )
        self.image = cv.imread(image_path, 1)

    def models_path(self, model_name):
        model_path = os.path.join(
            ASSETS_FOLDER,
            "models",
            model_name
        )
        return model_path

    def get_letters(self):
        words_coordinates = [
            np.array([[83, 74],[160, 240]]),
            np.array([[83, 283],[160, 453]]),
            np.array([[83, 508],[160, 640]]),
            np.array([[83, 672],[160, 845]]),
            np.array([[83, 876],[160, 1040]]),
            np.array([[83, 1068],[160, 1297]]),
        ]

        self.words = list()
        self.letters_words = list()

        for coor in words_coordinates:
            for j in iter(range(coor.shape[1])):
                coor[:,j] = np.sort(coor[:,j], 0)

            roiImage = self.image[
                coor[0,0]:coor[1,0],
                coor[0,1]:coor[1,1]
            ]

            self.words.append(roiImage)
        
        word_wei = [39, 41, 35, 41, 35, 45]

        for w in iter(range(len(self.words))):
            letters_buff = list()
            for i in iter(range(int(self.words[w].shape[1]/word_wei[w]))):
                roiImage_2 = self.words[w][:,i*word_wei[w]:(i+1)*word_wei[w]]
                letters_buff.append(roiImage_2)
            self.letters_words.append(letters_buff)

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

    def get_invariant_moments(self, M, cx, cy):
        # Calculate the invariant moments on translational transformations
        u11 = M['m11'] - cx*M['m01']
        u20 = M['m20'] - cx*M['m10']
        u02 = M['m02'] - cy*M['m01']
        u21 = M['m21'] - 2*cx*M['m11'] - cy*M['m20'] - 2*(cx**2)*M['m01']
        u12 = M['m12'] - 2*cy*M['m11'] - cx*M['m02'] - 2*(cy**2)*M['m10']
        u30 = M['m30'] - 3*cx*M['m20'] + 2*(cx**2)*M['m10']
        u03 = M['m03'] - 3*cy*M['m02'] + 2*(cy**2)*M['m01']

        # Calculate the invariant moments on small transformations
        I1 = (1/M['m00']**4)*(u20*u02 - u11**2)
        I2 = (1/M['m00']**10)*((-u30**2)*(u03**2) + 6*u30*u21*u12*u03 - 4*u30*(u12**3) - 4*(u21**3)*u03 + 3*(u21**2)*(u12**2))
        I3 = (1/M['m00']**7)*(u20*u21*u03 - u20*(u12**2) - u11*u30*u03 + u11*u21*u12 + u02*u30*u12 - u02*(u21**2))
        I4 = (1/M['m00']**11)*((-u20**3)*(u03**2) + 6*(u20**2)*u11*u12*u03 - 3*(u20**2)*u02*(u12**2) - 6*u20*(u11**2)*u21*u03\
            - 6*u20*(u11**2)*(u12**2) + 12*u20*u11*u02*u21*u12 - 3*u20*(u02**2)*(u21**2) + 2*(u11**3)*u30*u03 + 6*(u11**2)*u02*u30*u12\
            - 6*(u11**2)*u02*(u21**2) + 6*u11*(u02**2)*u30*u21 - (u02**3)*(u30**2))

        return I1, I2, I3, I4

    def extract_features(self, img):
        # Define kernels for the filters
        erode_kernel = cv.getStructuringElement(cv.MORPH_RECT , (3,3))
        dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))

        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
        blur = cv.medianBlur(gray,5)

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

                    # Calculate additional moments 
                    I1, I2, I3, I4 = self.get_invariant_moments(M, cx, cy)

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
                        I2,
                        I3,
                        I4,
                        y_mean,
                        x_mean,
                        y_mean_half_l,
                        x_mean_half_l,
                        y_mean_half_u,
                        x_mean_half_u,
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
        CHARACTERS_LIST = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "P",
            "R",
            "S",
            "T",
            "V",
            "X",
            "Z"
        ]

        self.get_letters()
        result_string = ""

        for word in self.letters_words:
            for letter in word:
                X = self.extract_features(letter)

                # Scaling the data
                X = self.ss_model.transform(X)

                # Make PCA analysis
                X = self.pca_model.transform(X)

                # Predicting the letter of the image
                result = self.mlp_model.predict(X)

                # Fulfill the string
                result_string += CHARACTERS_LIST[int(result[0])]
            result_string += " "

        # Put text in the image
        print(result_string)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(self.image,result_string,(82,233), font, 2.3,(255,255,255),2,cv.LINE_AA)
        cv.imshow("Predictive Sentense", self.image)
        cv.waitKey(0)
            



def main():
    mlp_implementation = MlpImplementation().mlp_implementation()

if __name__ == "__main__":
    sys.exit(main())


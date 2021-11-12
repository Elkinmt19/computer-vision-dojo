# Built-in imports 
import os
import math
import sys
from glob import glob

# External imports 
import numpy as np
import cv2 as cv
import xlsxwriter
from sklearn.preprocessing import StandardScaler # Data scaling
from sklearn import decomposition #PCA
import plotly.express as px
import pandas as pd 
import joblib

# My Own imports
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

class CharactersDatasetPreprocessing:
    def __init__(self, model_flag):
        # Model's flag in order to allow save the models
        self.model_flag = model_flag

        # Define the resulted features vector
        self.characters_features = np.array([])

        # Define the first cell of the excel file 
        self.row = 0
        self.col = 0

        # Counter that allows to move between the excel's columns 
        self.xlx_counter = 1

        # Create a new excel file with a new sheet to work with
        path_dataset_file = os.path.join(
            ASSETS_FOLDER, "xlsx", "letters_dataset.xlsx")
        self.workbook = xlsxwriter.Workbook(path_dataset_file)
        self.worksheet = self.workbook.add_worksheet('characters-data')
        self.worksheet_pca = self.workbook.add_worksheet('characters-data_pca')

    def get_character_paths(self, character):
        # Universal path depending of the extention of the image
        path_characters = os.path.join(
            ASSETS_FOLDER, "imgs", f"bike_characters/{character}/*.jpg")

        # Function to find the images of the dataset
        self.img_names = glob(path_characters)

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

    def character_data_preprocessing(self, character_class):
        erode_kernel = cv.getStructuringElement(cv.MORPH_RECT , (2,2))
        dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
        for fn in self.img_names:
            img = cv.imread(fn, 1)

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
                cv.imshow("img_resize", binary_image_res)
                cv.waitKey(1)

                # Calculate more features based on the corners founded by Harris algorithm
                corner_x, corner_y = self.corner_detection(binary_image_res)

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

                        # Calculate additional features
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

                        # Add the data to the resulted features vector
                        ys = self.characters_features
                        xs = VectorCarac.reshape((1, VectorCarac.shape[0]))
                        self.characters_features = np.vstack([ys, xs]) if ys.size else xs

                        # Save the data in the excel file
                        self.worksheet.write(self.row, self.col, character_class)
                        self.worksheet_pca.write(self.row, self.col, character_class)
                        for carac in (VectorCarac):
                            self.worksheet.write(self.row, self.xlx_counter, carac)
                            self.xlx_counter += 1
                        self.xlx_counter = 1
                        self.row += 1
                    except:
                        pass

    def principal_component_anaylis(self):
        # Data scaling
        ss = StandardScaler()
        X = ss.fit_transform(self.characters_features)

        # Perform the PCA analysis 
        pca = decomposition.PCA(n_components=30)
        pca.fit(X)

        # Calculate the scores values
        scores = pca.transform(X)

        # Save the data in the excel file
        for i in iter(range(scores.shape[0])):
            for j in iter(range(scores.shape[1])):
                self.worksheet_pca.write(i, j+1, scores[i,j])

        # Save important models
        if (self.model_flag):
            joblib.dump(ss, self.models_path("model_scaling_all_letters.pkl"))
            joblib.dump(pca, self.models_path("model_pca_all_letters.pkl"))

        # Explained variance for each PC
        explained_variance = pca.explained_variance_ratio_
        explained_variance = np.insert(explained_variance, 0, 0)

        # Combining the dataframe
        pc_df = [f"PC{x}" for x in iter(range(scores.shape[1] + 1))]
        pc_df = pd.DataFrame(pc_df, columns=['PC'])
        explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'])

        df_explained_variance = pd.concat([pc_df, explained_variance_df], axis=1)

        fig = px.bar(
            df_explained_variance, 
            x='PC', y='Explained Variance',
            text='Explained Variance',
            width=800
        )

        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.show()
    
    def models_path(self, model_name):
        model_path = os.path.join(
            ASSETS_FOLDER,
            "models",
            model_name
        )
        return model_path

    def complete_characters_dataset(self):
        character_list_numbers = [
            "0","1",
            "2","3",
            "4","5",
            "6","7",
            "8","9"
        ]

        character_list_letters = [
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

        for i in iter(range(len(character_list_letters))):
            # Get the number paths
            self.get_character_paths(character_list_letters[i])

            # Process the data of the character
            self.character_data_preprocessing(i)

        # Do the principal component analysis
        self.principal_component_anaylis()

        # Close the excel file
        self.workbook.close()
        cv.destroyAllWindows()


def main():
    characters_prepro = CharactersDatasetPreprocessing(True)
    characters_prepro.complete_characters_dataset()

if __name__ == "__main__":
    sys.exit(main())

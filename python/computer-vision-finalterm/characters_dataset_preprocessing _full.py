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
            ASSETS_FOLDER, "xlsx", "characters_dataset.xlsx")
        self.workbook = xlsxwriter.Workbook(path_dataset_file)
        self.worksheet = self.workbook.add_worksheet('characters-data')
        self.worksheet_pca = self.workbook.add_worksheet('characters-data_pca')

    def get_character_paths(self, character):
        # Universal path depending of the extention of the image
        path_characters = os.path.join(
            ASSETS_FOLDER, "imgs", f"numbers_letters/{character}/*.jpg")

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

            # Vector with the object's features
            VectorCarac = []

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

                        if(cv.contourArea(cnt_2)>100):
                            # Divided the image image
                            for j in range(0,4):  #Se dividió 20 y 40 por 5
                                for i in range(0,6):
                                    imgRoi_2 = img_resize[i*5:(i+1)*5, j*5:(j+1)*5] #Acá se estan haciento recortes de 5x5 a la imagen, como filtros
                                    hh,ww = imgRoi_2.shape[:2]
                                    for k in range(0,hh):
                                        # Count the white pixels of the peace of the image
                                        valRow = cv.countNonZero(imgRoi_2[k:k+1,:])                        
                                        diag = np.diagonal(imgRoi_2[k:k+1,:])                            
                                        val_diag = (diag[0])/255.0
                                        valRow = valRow/hh
                                        VectorCarac.append(valRow)
                                        VectorCarac.append(val_diag)
                                    for m in range(0,ww):
                                        #print("por columnas...")
                                        #cv.imshow("imgRoi_2_columnas",cv.resize(imgRoi_2[:,m:m+1],(10,50)))
                                        valCol = cv.countNonZero(imgRoi_2[:,m:m+1])
                                        diag = np.diagonal(imgRoi_2[:,m:m+1])
                                        val_diag = (diag[0])/255.0
                                        valCol = valCol/hh
                                        VectorCarac.append(valCol)
                                        VectorCarac.append(val_diag)

                        # Vector with the object's features
                        VectorCarac = np.array(
                            VectorCarac,
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
        # pca = decomposition.PCA(n_components=4)
        # pca.fit(X)

        # Calculate the scores values
        # scores = pca.transform(X)
        scores = ss.transform(X)

        # Save the data in the excel file
        for i in iter(range(scores.shape[0])):
            for j in iter(range(scores.shape[1])):
                self.worksheet_pca.write(i, j+1, scores[i,j])

        # Save important models
        if (self.model_flag):
            joblib.dump(ss, self.models_path("model_scaling_all_characters.pkl"))
            # joblib.dump(pca, self.models_path("model_pca_all_characters.pkl"))

        # Explained variance for each PC
        # explained_variance = ss.explained_variance_ratio_
        # explained_variance = np.insert(explained_variance, 0, 0)

        # # Combining the dataframe
        # pc_df = [f"PC{x}" for x in iter(range(scores.shape[1] + 1))]
        # pc_df = pd.DataFrame(pc_df, columns=['PC'])
        # explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'])

        # df_explained_variance = pd.concat([pc_df, explained_variance_df], axis=1)

        # fig = px.bar(
        #     df_explained_variance, 
        #     x='PC', y='Explained Variance',
        #     text='Explained Variance',
        #     width=800
        # )

        # fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        # fig.show()
    
    def models_path(self, model_name):
        model_path = os.path.join(
            ASSETS_FOLDER,
            "models",
            model_name
        )
        return model_path

    def complete_characters_dataset(self):
        character_list = [
            "0","1",
            "2","3",
            "4","5",
            "6","7",
            "8","9",
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
        for i in iter(range(len(character_list))):
            # Get the number paths
            self.get_character_paths(character_list[i])

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

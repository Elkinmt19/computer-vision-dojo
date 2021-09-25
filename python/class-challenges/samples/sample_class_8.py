# Built-in imports 
import sys
import os

# External imports 
import numpy as np
import cv2 as cv
import xlrd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import joblib

# My Own imports
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

# Create a new excel file with a new sheet to work with
path_dataset_file = os.path.join(
    ASSETS_FOLDER, "xlsx", "complete_numbers_dataset.xlsx")
workbook = xlrd.open_workbook(path_dataset_file)

def load_excel(xlsx):
    worksheet = xlsx.sheet_by_index(0)
    X = np.zeros((worksheet.nrows, worksheet.ncols - 1))
    Y = []

    for i in iter(range(worksheet.nrows)):
        for j in iter(range(worksheet.ncols - 1)):
            X[i,j] = worksheet.cell_value(rowx=i, colx=j+1)
            
        Y.append(worksheet.cell_value(rowx=i, colx=0))
        
    Y = np.array(Y, np.float32)
    return X, Y

def mlp_model():
    X, Y = load_excel(workbook)

    ss = StandardScaler()
    X = ss.fit_transform(X)

    samples_train, samples_test, responses_train, responses_test = train_test_split(X, Y, test_size=0.3)

    mlp = MLPClassifier(activation="relu", hidden_layer_sizes=(100,100), max_iter=1000, tol=0.0001)

    mlp.fit(samples_train, responses_train)
    result_accuracy = accuracy_score(responses_test, mlp.predict(samples_test))
    print(result_accuracy*100.0)

    def save_models(model_name):
        # Save the models
        path_dataset_file = os.path.join(
            ASSETS_FOLDER, "models", model_name)
        return path_dataset_file

    joblib.dump(ss, save_models("model_ss.pkl"))
    joblib.dump(mlp, save_models("model_mlp.pkl"))

def svm_model():
    X, Y = load_excel(workbook)

    ss = StandardScaler()
    X = ss.fit_transform(X)
    vector_C = [1, 10, 100]

    for i in iter(range(3)):
        samples_train, samples_test, responses_train, responses_test = train_test_split(X, Y, test_size=0.3)

        svm = SVC(C=vector_C[i], kernel="rbf")

        svm.fit(samples_train, responses_train)
        result_accuracy = accuracy_score(responses_test, svm.predict(samples_test))
        print(result_accuracy*100.0)

        def save_models(model_name):
            # Save the models
            path_dataset_file = os.path.join(
                ASSETS_FOLDER, "models", model_name)
            return path_dataset_file

        joblib.dump(ss, save_models("model_ss.pkl"))
        joblib.dump(svm, save_models("model_svm.pkl"))

def main():
    # mlp_model()
    svm_model()

if __name__ == "__main__":
    sys.exit(main())

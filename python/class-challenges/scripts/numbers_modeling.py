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

class ModelTraining:
    """
    This is a python class that implements two models of machine learning 
    MLP (Multi Layer Perceptron) and SVM (Support Vector Machine), in order 
    to get a good model based on a dataset of a serie of numbers.

    :param model: model that is gonna be used {mlp or svm} (String).
    :param model_params: list of params of the model that was choosen (List).
    """
    def __init__(self, model, model_params):
        self.model = model 
        self.model_params = model_params

        # Create a new excel file with a new sheet to work with
        path_dataset_file = os.path.join(
            ASSETS_FOLDER,
            "xlsx", "complete_numbers_dataset.xlsx"
        )
        self.workbook = xlrd.open_workbook(path_dataset_file)

    def load_data(self):
        """
        This is a python method that allows to get the data of a dataset that
        is saved in an exce file, this function takes the information of the inputs 
        and the desired variables for the training of the model.
        """
        self.worksheet = self.workbook.sheet_by_index(1)
        X = np.zeros((self.worksheet.nrows, self.worksheet.ncols - 1))
        Y = []

        for i in iter(range(self.worksheet.nrows)):
            for j in iter(range(self.worksheet.ncols - 1)):
                X[i,j] = self.worksheet.cell_value(rowx=i, colx=j+1)
                
            Y.append(self.worksheet.cell_value(rowx=i, colx=0))
            
        Y = np.array(Y, np.float32)
        return X, Y

    def models_path(self, model_name):
        model_path = os.path.join(
            ASSETS_FOLDER,
            "models",
            model_name
        )
        return model_path

    def mlp_model(self):
        """
        This is a python method that implements a MLP (Multi Layer Perceptron)
        model based on a dataset that was preprocessed, this method used the sklearn 
        python module in order to achive this task.
        """
        # Get the data of the training of the model
        X, Y = self.load_data()

        # Split the data in order to train and test the model
        samples_train, samples_test, responses_train, responses_test = train_test_split(X, Y, test_size=0.3)

        # Build the model
        mlp = MLPClassifier(
            activation=self.model_params[0],
            hidden_layer_sizes=self.model_params[1],
            max_iter=self.model_params[2],
            tol=self.model_params[3]
        )

        # Train the model based on the corresponding parameters
        mlp.fit(samples_train, responses_train)
        result_accuracy = accuracy_score(responses_test, mlp.predict(samples_test))
        print(f"Resulting Accuracy: {result_accuracy*100.0}")

        # Save the resulting models
        joblib.dump(X, self.models_path("model_Xmpl_all_numbers.pkl"))
        joblib.dump(mlp, self.models_path("model_mlp_all_numbers.pkl"))

    def svm_model(self):
        """
        This is a python method that implements a SVM (Support Vector Machine)
        model based on a dataset that was preprocessed, this method used the sklearn 
        python module in order to achive this task.
        """
        # Get the data of the training of the model
        X, Y = self.load_data()

        # Split the data in order to train and test the model
        samples_train, samples_test, responses_train, responses_test = train_test_split(X, Y, test_size=0.3)

        # Build the model
        svm = SVC(
            C=self.model_params[0],
            kernel=self.model_params[1]
        )

        # Train the model based on the corresponding parameters
        svm.fit(samples_train, responses_train)
        result_accuracy = accuracy_score(responses_test, svm.predict(samples_test))
        print(f"Resulting Accuracy: {result_accuracy*100.0}")

        # Save the resulting models
        joblib.dump(X, self.models_path("model_Xsvm_all_numbers.pkl"))
        joblib.dump(svm, self.models_path("model_svm_all_numbers.pkl"))

    def train_model(self):
        try:
            if (self.model == "mlp"):
                self.mlp_model()
            if (self.model == "svm"):
                self.svm_model()
        except:
            print("A PROBLEM HAPPEN!!!")

def main():
    mlp_params = [
        "relu",
        (100,100),
        1000,
        0.0001
    ]

    svm_params = [
        1,
        "rbf"
    ]

    model_train = ModelTraining("mlp", mlp_params)
    for _ in iter(range(10)):
        model_train.train_model()


if __name__ == "__main__":
    sys.exit(main())

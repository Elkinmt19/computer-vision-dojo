# Built-in imports 
import sys
import os

# External imports 
import joblib

# My own imports 
import get_path_assests_folder as gpaf
import characters_modeling as cmlp

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

class PipelineMLP:
    def __init__(self):
        self.base_params = [
            None,
            None,
            5000,
            0.0001
        ]
        self.pipeline = {
            "hidden-layers": [
                (100,),
                (100,500)
            ],
            "activation-function": [
                "relu",
                "tanh"
            ]
        }

    def models_path(self, model_name):
        model_path = os.path.join(
            ASSETS_FOLDER,
            "models",
            model_name
        )
        return model_path

    def execute_pipeline(self):
        best_model = [None, None]
        best_accuracy = 0
        for hl in self.pipeline["hidden-layers"]:
            for af in self.pipeline["activation-function"]:
                model_description = f"MLP model with function {af} and {len(hl)} hidden layers of {hl} neurons"
                print("Calculating " + model_description + "...")

                self.base_params[0] = af
                self.base_params[1] = hl

                model_train = cmlp.ModelTraining(
                    "mlp",
                    self.base_params,
                    False,
                    False
                )
                mlp, accuracy = model_train.mlp_model()

                if (accuracy > best_accuracy):
                    best_model[0] = model_description
                    best_model[1] = mlp

                    best_accuracy = accuracy

        print("The best model is " + best_model[0])
        print("Saving the model...")
        joblib.dump(mlp, self.models_path("model_mlp_characters.pkl"))


def main():
    pipeline = PipelineMLP().execute_pipeline()


if __name__ == "__main__":
    sys.exit(main())

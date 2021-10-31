""" SCRIPT TO GET ASSETS FOLDER BASED ON REPOSITORY STRUCTURE """
# Made by https://github.com/san99tiago

# Built-int imports
import os


def get_assets_folder_path():
    # Assets folder is located 1 directories up (double parent dir)
    COMPUTER_VISION_FOLDER = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.path.pardir
        )
    )

    ASSETS_FOLDER_PATH = os.path.abspath(
        os.path.join(COMPUTER_VISION_FOLDER, "assets")
    )
    return ASSETS_FOLDER_PATH


if __name__ == "__main__":
    print(get_assets_folder_path())

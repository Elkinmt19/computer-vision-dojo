""" SCRIPT TO GET ASSETS FOLDER BASED ON REPOSITORY STRUCTURE """
# Made by https://github.com/san99tiago

# Built-int imports
import os


def get_assets_folder_path():
    # Assets folder which is in the same level of the file
    COMPUTER_VISION_FOLDER = os.path.abspath(
        os.path.dirname(__file__)
    )

    ASSETS_FOLDER_PATH = os.path.abspath(
        os.path.join(COMPUTER_VISION_FOLDER, "assets")
    )
    return ASSETS_FOLDER_PATH


if __name__ == "__main__":
    print(get_assets_folder_path())

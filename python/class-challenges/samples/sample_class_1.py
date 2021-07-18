# Built-int imports
import os

# External imports 
import cv2 as cv

# My Own imports
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()


def load_image(path, mode):
    return (cv.imread(path, mode))

def show_image(nameWin, img):
    cv.imshow(nameWin, img)

def closeWin():
    cv.destroyAllWindows()

def get_pixel_image(img):
    rows, cols = img.shape[:2]
    for i in range(rows):
        for j in range(cols):
            print(img[i,j])

def main():
    image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", "Tony_Stark.jpeg")
    colorImage = load_image(image_relative_path, 1)
    show_image("Tony Stark", colorImage)
    #get_pixel_image(colorImage)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
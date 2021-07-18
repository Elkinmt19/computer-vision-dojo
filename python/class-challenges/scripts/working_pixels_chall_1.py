# Built-int imports 
import os

# External imports
import cv2 as cv
import numpy as np

# My own imports 
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()


class DolphinPlayingWithPixels:
    def __init__(self):
        self.__delfin_path = self.get_image_path()
        self.image = cv.imread(self.__delfin_path, 1)
        self.__rows, self.__cols, self.__channels = self.image.shape


    def get_image_path(self):
        image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", "delfin.jpg")
        return image_relative_path

    def dolphin_separated(self, channel):
        new_image = cv.imread(self.__delfin_path, 1)
        for (i,j,c), _ in np.ndenumerate(new_image):
            if c == channel:
                new_image[i,j,c] = 255
        cv.imshow(f"New dolphin channel: {channel}", new_image)
    
    def dolphin_channels(self):
        cv.imshow("Original dolphin", self.image)
        for i in iter(range(3)):
            self.dolphin_separated(i)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def dolphin_y_reverse(self):
        new_image = cv.imread(self.__delfin_path, 1)

        for i in iter(range(self.__rows)):
            for c in iter(range(self.__channels)):
                for j in iter(range(self.__cols)):
                    new_image[i,j,c] = self.image[i,(self.__cols-1)-j, c]
        cv.imshow("Original dolphin", self.image)
        cv.imshow("y reverse dolphin", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def dolphin_x_reverse(self):
        new_image = cv.imread(self.__delfin_path, 1)

        for j in iter(range(self.__cols)):
            for c in iter(range(self.__channels)):
                for i in iter(range(self.__rows)):
                    new_image[i,j,c] = self.image[(self.__rows-1)-i, j, c]
        cv.imshow("Original dolphin", self.image)
        cv.imshow("x reverse dolphin", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def dolphin_colorfull(self):
        new_image = cv.imread(self.__delfin_path, 1)

        for (i,j,c), _ in np.ndenumerate(new_image):
            if (i < self.__rows/2 and j >= self.__cols/2 and c == 0): 
                new_image[i,j,c] = 255
            if (i >= self.__rows/2 and j < self.__cols/2 and c == 1): 
                new_image[i,j,c] = 255
            if (i >= self.__rows/2 and j >= self.__cols/2 and c == 2): 
                new_image[i,j,c] = 255
        cv.imshow("Original dolphin", self.image)
        cv.imshow("Colorfull dolphin", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def dolphin_colorfull_by_user(self, px, py):
        new_image = cv.imread(self.__delfin_path, 1)
        
        x_count , y_count, c_count = (0,0,0)
        x_refe, y_refe = (0,0)
        x_number = int(self.__rows/px)
        y_number = int(self.__cols/py)
        print(f"x_number: {x_number} y_number: {y_number}")

        for i in iter(range(self.__rows)):
            x_count = i - x_refe
            for j in iter(range(self.__cols)):
                y_count = j - y_refe
                for c in iter(range(self.__channels)):
                    if (x_count <= px and y_count <= py and c == c_count):
                        new_image[i,j,c] = 255
                    elif (x_count > px):
                        x_refe = i
                    elif (y_count > py):
                        y_refe = j
                        
        cv.imshow("Original dolphin", self.image)
        cv.imshow("Colorfull by user dolphin", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()




def main():
    dolphin = DolphinPlayingWithPixels()
    dolphin.dolphin_colorfull_by_user(150,150)


if __name__=='__main__':
    main()
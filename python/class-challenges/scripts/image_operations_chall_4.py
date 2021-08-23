# Built-int imports 
import os
import sys
import math
import argparse

# External imports
import cv2 as cv
import numpy as np

# My own imports 
import image_segmentation as imgs
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

class ImageOperations:
    """
    This is a python class which allows to analyze the images
    of a product of pills and inpect if the product has any kind
    of imperfection and also allows to get the velocity of a car and
    a truck by some images took by a traffic camera.
    """
    def __init__(self):
        self.get_images()

    def get_image_path(self, image_name):
        image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", image_name)
        return image_relative_path

    def get_images(self):
        # Upload the images for the first point
        self.pills = [
            cv.imread(self.get_image_path(f"pills_{x}.png")) for x in iter(range(1,3))
        ]

        # Upload the images for the second point
        self.frames = [
            cv.imread(self.get_image_path(f"frame_{x}.png")) for x in iter(range(1,31,29))
        ]

    def get_values_pills_segmentation(self):
        """
        This is a simple method to find the umbral's values 
        for the segmentation of the pill images for the first
        point of the challenge.
        """
        for img in self.pills:
            color_seg = imgs.ImageSegmentation(img)
            color_seg.color_segmentation("HSV")

    def __get_location(self, x, y, vx, vy):
        x_bool = list()
        y_bool = list()
        for p_range in x:
            x_bool.append(vx > p_range[0] and vx < p_range[1])

        for p_range in y:
            y_bool.append(vy > p_range[0] and vy < p_range[1])

        return (x_bool.index(True), y_bool.index(True))
    
    def count_pills(self):
        """
        This is a method which allows to count the number of pills 
        That a product of the company has, and knows when pills are missing
        and also says the location of these ones.
        """
        segmented_pills = list()
        upper_values = np.array([255, 193, 255])
        lower_values = np.array([0, 100, 137])

        for img in self.pills:
            segmented_pills.append(cv.inRange(
                cv.cvtColor(img, cv.COLOR_BGR2HSV),
                lower_values,
                upper_values
            ))
        
        subtracted_pill = cv.subtract(
            segmented_pills[0],
            segmented_pills[1]
        )

        blur_image = cv.medianBlur(subtracted_pill, 15)
        num_labels, labels_img = cv.connectedComponents(blur_image)
        print(f"Number of labels: {num_labels}")

        x_coor = [(40,183),(220,377),(421,566)]
        y_coor = [(53,98),(126,170),(203,249),(277,327)]

        for i in iter(range(1,num_labels)):
            index_element = np.where(labels_img == i)
            my = sum(index_element[0])/index_element[0].shape[0]
            mx = sum(index_element[1])/index_element[1].shape[0]
            print("Imperfection detected - A pill is missing")
            print(f"Mean Index: mx = {mx} my = {my}")
            location  = self.__get_location(x_coor, y_coor, mx, my)
            print(f"The location is : {location}")

        print(f"The number of pills of this product is : {13 - num_labels}")
        cv.imshow("Subtracted pill", blur_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def get_values_frames_segmentation(self):
        """
        This is a simple method to find the umbral's values 
        for the binarization of the frame images for the second
        point of the challenge.
        """

        for img in self.frames:
            color_seg = imgs.ImageSegmentation(img)
            color_seg.color_segmentation("HSV")

    def get_x_distance_between_cars(self):
        """
        This is a method to find the x-distances and velocities
        of the car and the truck showed in the images took by 
        a traffic camera.
        """
        binarized_car = list()
        binarized_truck = list()

        upper_values_car = np.array([152, 160, 236])
        lower_values_car = np.array([102, 89, 58])

        upper_values_truck = np.array([115, 255, 255])
        lower_values_truck = np.array([4, 134, 214])

        for img in self.frames:
            binarized_car.append(cv.inRange(
                cv.cvtColor(img, cv.COLOR_BGR2HSV),
                lower_values_car,
                upper_values_car
            ))
        
        for img in self.frames:
            binarized_truck.append(cv.inRange(
                cv.cvtColor(img, cv.COLOR_BGR2HSV),
                lower_values_truck,
                upper_values_truck
            ))
        
        sum_car = cv.add(
            binarized_car[0],
            binarized_car[1]
        )
        sum_car = cv.medianBlur(sum_car, 7)
        num_labels_car, labels_img_car = cv.connectedComponents(sum_car)
        print(f"Number of labels car: {num_labels_car}")

        sum_truck = cv.add(
            binarized_truck[0],
            binarized_truck[1]
        )
        sum_truck = cv.medianBlur(sum_truck, 7)
        num_labels_truck, labels_img_truck = cv.connectedComponents(sum_truck)
        print(f"Number of labels truck: {num_labels_truck}")

        time = 30.0/45.0
        mx_car = list()
        for i in iter(range(1,num_labels_car)):
            index_element = np.where(labels_img_car == i)
            mx_car.append(sum(index_element[1])/index_element[1].shape[0])
        x_distance_car = math.fabs(mx_car[1]-mx_car[0])
        print(f"The x-distance of the car is {x_distance_car} pixels")
        print(f"The x-speed of the car is {x_distance_car/time} pixel/seg")

        mx_truck = list()
        for i in iter(range(1,num_labels_truck)):
            index_element = np.where(labels_img_truck == i)
            mx_truck.append(sum(index_element[1])/index_element[1].shape[0])
        x_distance_truck = math.fabs(mx_truck[1]-mx_truck[0])
        print(f"The x-distance of the truck is {x_distance_truck} pixels")
        print(f"The x-speed of the car is {x_distance_truck/time} pixel/seg")

        cv.imshow("Car sum", sum_car)
        cv.imshow("Truck sum", sum_truck)

        cv.waitKey(0)
        cv.destroyAllWindows()



def main():
    """COMPUTER VISION - EIA UNIVERSITY
    Fourth challenge of the EIA University's computer vision class.
    Run this scripts in order to see Elkin Guerra's solucion 
    of this test. 
    """
    epilog = """
    Related examples:
    More to come...
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
        description=main.__doc__,
        epilog=epilog
    )
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-s', '--stage', dest='stage', required=True, choices=[
            "one","two"
        ],
        help='The stage of the challenge you want to execute'
    )

    args = parser.parse_args()

    print("Initializing program... ")
    img_operator = ImageOperations()

    try:
        act = args.stage
        if act == "one":
            img_operator.count_pills()
        elif act == "two":
            img_operator.get_x_distance_between_cars()
            
    except:
        print("ERROR JUST HAPPEND")

    return 0

if __name__=='__main__':
    sys.exit(main())
# Built-int imports
import sys
import argparse

# External imports 
import cv2 as cv
import numpy as np

# Own imports 
import image_segmentation as imgse


class ComputerVisionQuiz1TopicOne:
    def __init__(self):
        self.generate_image_list()

    def generate_image_list(self):
        print("Loading original images in a list...")
        path_ori_imgs = "./first-question/originales/"
        self.originals_images = [cv.imread(f"{path_ori_imgs}placa_{x+1}_Original.png") for x in iter(range(5))]

        print("Loading wrong images in a list...")
        path_wrong_imgs = "./first-question/malas/"
        self.wrong_images = [cv.imread(f"{path_wrong_imgs}placa_{x+1}_P1.png") for x in iter(range(5))]

    def validate_x_flip(self):
        wrong_images_copy = self.wrong_images.copy()
        validation_list = list()

        for i in iter(range(len(self.originals_images))):
            validation_list.append((cv.flip((wrong_images_copy[i]),0)==self.originals_images[i]).all())

        return validation_list

    def validate_y_flip(self):
        wrong_images_copy = self.wrong_images.copy()
        validation_list = list()

        for i in iter(range(len(self.originals_images))):
            validation_list.append((cv.flip((wrong_images_copy[i]),1)==self.originals_images[i]).all())

        return validation_list

    def validate_pi_rotation(self):
        wrong_images_copy = self.wrong_images.copy()
        validation_list = list()

        for i in iter(range(len(self.originals_images))):
            validation_list.append((cv.flip((wrong_images_copy[i]),-1)==self.originals_images[i]).all())

        return validation_list

    def compare_images(self):
        print("Comparing lists of images...")
        list_x_validation = self.validate_x_flip()
        list_y_validation = self.validate_y_flip()
        list_pi_rotation = self.validate_pi_rotation()

        print("Fixing images...")
        # Fix the x flip images
        for i in iter(range(len(list_x_validation))):
            if list_x_validation[i]:
                self.wrong_images[i] = cv.flip(self.wrong_images[i], 0)

        # Fix the y flip images
        for i in iter(range(len(list_y_validation))):
            if list_y_validation[i]:
                self.wrong_images[i] = cv.flip(self.wrong_images[i], 1)

        # Fix the pi rotated images
        for i in iter(range(len(list_pi_rotation))):
            if list_pi_rotation[i]:
                self.wrong_images[i] = cv.flip(self.wrong_images[i], -1)

        print("Images fixed!")
        print("Saving fixed images in './first-question/Fixed_images/' path...")
        # Saving fixed images
        for i in iter(range(len(self.wrong_images))):
            cv.imwrite(f"./first-question/Fixed_images/placa_{i+1}_fixed.png", self.wrong_images[i])

    def get_values_color_segmentation(self):
        for img in self.wrong_images:
            color_seg = imgse.ImageSegmentation(img)
            color_seg.color_segmentation("HSV")

    def segmentation_condition(self, img, i , j, umbral):
        return (img[i,j,0] > umbral[0][0] and img[i,j,0] < umbral[0][1] and \
        img[i,j,1] > umbral[1][0] and img[i,j,1] < umbral[1][1] and \
        img[i,j,2] > umbral[2][0] and img[i,j,2] < umbral[2][1])

    def color_segmentation(self):
        # Umbral in HSV for each image
        list_umbrals = [
            [(145,217),(124,255),(0,197)],
            [(0,194),(55,207),(74,152)],
            [(50,128),(56,255),(0,133)],
            [(0,205),(38,255),(27,164)],
            [(0,227),(0,128),(17,58)]
        ]
        
        print("Segmentating images...")
        for w in iter(range(len(self.wrong_images))):
            image = self.wrong_images[w].copy()
            hsv_image = cv.cvtColor(self.wrong_images[w], cv.COLOR_RGB2HSV)
            for i in iter(range(self.wrong_images[w].shape[0])):
                for j in iter(range(self.wrong_images[w].shape[1])):
                    if self.segmentation_condition(hsv_image,i,j,list_umbrals[w]):
                        image[i,j] = 255
                    else:
                        image[i,j] = 0
            
            cv.imwrite(f"./first-question/Segmentated_images/placa_{w+1}_segmentated.png", image)
        print("Segmentated images saved in './first-question/Segmentated_images/' path...")


class ComputerVisionQuiz1TopicTwo:
    def __init__(self):
        self.get_medical_image()
        self.coordenates = np.array([[0,0],[0,0]])
        self.print_flag = False

    def get_medical_image(self):
        image_name = input("Enter the name of the image you wanna analyze: ")
        self.medical_image = cv.imread(f"./second-question/{image_name}")

    def click_mouse_callback(self, event, y, x, flags, param):
        """
        Click-mouse callback function to use a click event
        """
        if (event == cv.EVENT_LBUTTONDOWN):
            self.coordenates[0,0] = x
            self.coordenates[0,1] = y

        if (event == cv.EVENT_LBUTTONUP):
            self.coordenates[1,0] = x
            self.coordenates[1,1] = y
            self.print_flag = True

    def get_region_of_interest(self):
        cv.namedWindow("Medical image")
        cv.setMouseCallback("Medical image", self.click_mouse_callback)

        print("Getting the region of interest...")
        while (True):
            cv.imshow("Medical image", self.medical_image)
            if self.print_flag:
                print(f"Location: ({self.coordenates})")

                #! Fix the problem with negative-coordinates
                for j in iter(range(self.coordenates.shape[1])):
                    self.coordenates[:,j] = np.sort(self.coordenates[:,j], 0)

                self.roiImage = self.medical_image[
                    self.coordenates[0,0]:self.coordenates[1,0],
                    self.coordenates[0,1]:self.coordenates[1,1]
                ]

                self.roiImage = cv.resize(
                    self.roiImage,
                    [x*10 for x in self.roiImage.shape[:2]][::-1]
                )

                cv.imshow("Medical image - ROI",self.roiImage)
                self.print_flag = False
            if (cv.waitKey(1) & 0xFF == ord('q')):
                break

    def  get_values_binarization(self):
        color_seg = imgse.ImageSegmentation(self.roiImage)
        color_seg.binarization()

    def binarization(self):
        image = self.roiImage.copy()
        gray_image = cv.cvtColor(self.roiImage, cv.COLOR_RGB2GRAY)

        print("Binarizing image...")
        for i in iter(range(self.roiImage.shape[0])):
            for j in iter(range(self.roiImage.shape[1])):
                if (gray_image[i,j] > 144 and gray_image[i,j] < 167):
                    image[i,j] = 255
                else:
                    image[i,j] = 0
        cv.imwrite(f"./second-question/fractura_binarized.jpg", image)
        print("Binarized image saved in './second-question/' path...")

def topic_one():
    cv_quiz1 = ComputerVisionQuiz1TopicOne()
    cv_quiz1.compare_images()
    cv_quiz1.color_segmentation()

def topic_two():
    cv_quiz1 = ComputerVisionQuiz1TopicTwo()
    cv_quiz1.get_region_of_interest()
    cv_quiz1.binarization()

def main():
    """COMPUTER VISION - EIA UNIVERSITY
    Quiz one of the EIA University's computer vision class
    Authors:
        :Elkin Javier Guerra Galeano
        :Sebastian Andres Padilla Mendoza
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
        help='The stage of the quiz that you want to run'
    )

    args = parser.parse_args()

    print("Initializing program... ")

    try:
        act = args.stage
        if act == "one":
            topic_one()
        elif act == "two":
            topic_two()        
    except:
        print("ERROR JUST HAPPEND")

    return 0

if __name__ == "__main__":
    sys.exit(main())

# Built-in imports 
import sys
import math

# External imports
import cv2 as cv
import numpy as np

# My own imports 
import image_segmentation as imgs

class PoolGame:
    """
    This is a python class that allows to result the Midterm of the
    EIA University's Computer vision class, this class has methods that
    allows to analyze a pool game, segmentating the holes in the table 
    and the white ball of the game.
    """
    def __init__(self):
        self.video = cv.VideoCapture(self.get_video_path("billar.mp4"))
        self.get_images()

    def get_video_path(self, video_name):
        video_relative_path = video_name
        return video_relative_path

    def get_images(self):
        self.images = list()
        self.images.append(cv.imread("holes.png",1))
        self.images.append(cv.imread("ball_2.png",1))

    def watch_video(self):
        """
        This is a simple method that allows to watch in a fast way 
        the video that we are working with.
        """
        while(self.video.isOpened()):
            ret, frame = self.video.read()
            if (ret == False):
                print("There is no image, the video is stoped")
                break
            cv.imshow("Frame", frame)
            if (cv.waitKey(20) & 0xFF == ord('q')):
                break

        self.video.release()
        cv.destroyAllWindows()

    def get_umbral_values_holes_segmentation(self):
        """
        This is a simple method that allows to find the perfect umbral's 
        values for the segmentation of the holes of the table.
        @ This method uses a class that is developed in the file 'image_segmentation.py'
        made by me.
        """
        img_seg = imgs.ImageSegmentation(self.images[0]).binarization()

    def holes_segmentation(self, img):
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _,binary_image = cv.threshold(gray_image, 27, 255, cv.THRESH_BINARY_INV)

        erode_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE , (20,20))
        dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20,20))

        # erode filter - to erode an image
        erode_image = cv.erode(binary_image, erode_kernel, iterations=1)

        # dilate filter - to dilate an image
        dilate_image = cv.dilate(erode_image, dilate_kernel, iterations=1)

        # Find the contours of the image
        contours, hier = cv.findContours(
            dilate_image.copy(),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE
        )

        return contours

    def get_umbral_values_balls_segmentation(self):
        """
        This is a simple method that allows to find the perfect umbral's 
        values for the segmentation of the white ball.
        @ This method uses a class that is developed in the file 'image_segmentation.py'
        made by me.
        """
        image = cv.resize(self.images[1],(700,500))
        img_seg = imgs.ImageSegmentation(image).color_segmentation("HSV")

    def ball_segmentation(self, img):
        hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # Umbral's values for the segmentation of the objects
        LOWER_VALUES = np.array([0, 0, 136])
        UPPER_VALUES = np.array([69, 53, 255])

        binary_image = cv.inRange(hsv_image, LOWER_VALUES, UPPER_VALUES)

        erode_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE , (6,6))
        dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (16,16))

        # medianBlur filter - to smooth out an image
        blur_image = cv.medianBlur(binary_image, 17)

        # erode filter - to erode an image
        erode_image = cv.erode(blur_image, erode_kernel, iterations=1)

        # dilate filter - to dilate an image
        dilate_image = cv.dilate(erode_image, dilate_kernel, iterations=1)

        # Find the contours of the image
        contours, hier = cv.findContours(
            dilate_image.copy(),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE
        )

        return contours

    def hole_number_validation(self, center):
        """
        This is a simple mthod that allows to map between the center of 
        each of the table's holes and the number that each of them have.
        """
        holes_centers = [
            [25, 23],
            [326, 18],
            [627, 23],
            [25, 322],
            [326, 323],
            [627, 322]
        ]

        for i in iter(range(len(holes_centers))):
            x_condition = (center[0] > (holes_centers[i][0] - 10)) and (center[0] < (holes_centers[i][0] + 10))
            y_condition = (center[1] > (holes_centers[i][1] - 10)) and (center[1] < (holes_centers[i][1] + 10))

            if (x_condition and y_condition):
                return (i + 1)

    def pool_game(self):
        """
        This is a method that uses the functionalities of the methods above
        in order to analyze the 'billar.mp4' video, this method gets the holes 
        of the table an their locations, it also gets the position of the 
        white ball in every time and with all this information this method 
        tells the user when exactly this ball has been inserted in which one 
        of the holes of the table.
        """
        ball_center = (0,0)
        holes_centers = list()
        while(self.video.isOpened()):
            ret, frame = self.video.read()
            if (ret == False):
                print("There is no image, the video is stoped")
                break
            holes_contours = self.holes_segmentation(frame)
            ball_contours = self.ball_segmentation(frame)

            # Draw the comtours in the image
            contour_image = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

            # Analysis of the contours of the table's holes
            for cnt in holes_contours:
                area = cv.contourArea(cnt)
                ratio = int(math.sqrt(area/math.pi))
                if (area > 0):
                    M = cv.moments(cnt)
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    cv.circle(contour_image, (cx,cy), ratio, (0,0,255), 2)
                    cv.circle(contour_image, (cx,cy), 1, (0,255,0), 2)

                    holes_centers.append([cx,cy])
            
            # Analysis of the contour of the white ball
            for cnt in ball_contours:
                area = cv.contourArea(cnt)
                ratio = int(math.sqrt(area/math.pi))
                if (area > 0):
                    M = cv.moments(cnt)
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    cv.circle(contour_image, (cx,cy), ratio, (0,255,0), 2)
                    cv.circle(contour_image, (cx,cy), 1, (255,0,0), 2)

                    ball_center = [cx,cy]

            for center in holes_centers:
                x_condition = (ball_center[0] > (center[0] - 3)) and (ball_center[0] < (center[0] + 3))
                y_condition = (ball_center[1] > (center[1] - 3)) and (ball_center[1] < (center[1] + 3))
                if (x_condition and y_condition):
                    print(f"Error - the white ball has been inserted in the hole {self.hole_number_validation(center)}")
                    ball_center = [0,0]


            cv.imshow("Frame", contour_image)
            if (cv.waitKey(1) & 0xFF == ord('q')):
                break

        self.video.release()
        cv.destroyAllWindows()



def main():
    """COMPUTER VISION - EIA UNIVERSITY
    Midterm of the EIA University's computer vision class.
    Run this scripts in order to see Elkin Javier Guerra
    Galeano's solucion of this test. 
    """

    # This is the main function of the whole script, just running this 
    # single command the application is gonna work.
    pool_game = PoolGame().pool_game()

if __name__ == "__main__":
    sys.exit(main())
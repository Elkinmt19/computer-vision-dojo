# Built-in imports 
import sys 

# External imports 
import cv2 as cv
import numpy as np
from numpy.lib.index_tricks import index_exp

# Own imports 
import sim
import kuka_youbot_class as kb

class ConveyorChallenge:
    def __init__(self):
        self.start_conextion()
        self.define_cube_camera()

        self.robot = kb.KukaYouBotClass(self.__clientID)

    def start_conextion(self):
        # End connexion 
        sim.simxFinish(-1)

        # Create new connexion
        self.__clientID = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)

        if (self.__clientID != -1):
            print("Connexion OK")
        else:
            print("Fatal error - No connexion")

    def stop_conextion(self):
        # End connexion 
        sim.simxFinish(-1)

    def define_cube_camera(self):
        # Define the camera's parameters
        _,self.cube_camera = sim.simxGetObjectHandle(
            self.__clientID,
            "Vision_sensor_cubes",
            sim.simx_opmode_blocking
        )

        _, self.resolution, self.image = sim.simxGetVisionSensorImage(
            self.__clientID,
            self.cube_camera,
            0,
            sim.simx_opmode_streaming
        )

    def cube_camera_buffer(self):
        """
        This is a method which allows to get an image from the 
        visual sensor of the CoppeliaSim Edu software.
        """
        _, self.resolution, self.image = sim.simxGetVisionSensorImage(
            self.__clientID,
            self.cube_camera,
            0,
            sim.simx_opmode_buffer
        )

    def get_image_cube_camera(self):
        """
        This is a method which counts the number of cubes in the table, 
        detects and segments the objects according their color and also 
        verify the quality of these objects. 
        """
        number_cubes = [0,0,0,0]
        faulty_cubes = [0,0,0,0]

        # Preprocessing of the image
        img = np.array(self.image, dtype = np.uint8)
        img.resize([self.resolution[0], self.resolution[1], 3])
        img = np.rot90(img,2)
        img = np.fliplr(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        # Umbral's values for the segmentation of the objects
        LOWER_VALUES = np.array(
            [
                [0, 142, 114],
                [125, 0, 0],
                [0, 0, 132],
                [0, 130, 0]
            ]
        )
        UPPER_VALUES = np.array(
            [
                [50, 252, 255],
                [255, 164, 129],
                [97, 77, 255],
                [62, 255, 131]
            ]
        )

        for i in iter(range(4)):
            binary_image = cv.inRange(img, LOWER_VALUES[i], UPPER_VALUES[i])

            # Find the contours of the image
            contours, hier = cv.findContours(
                binary_image.copy(),
                cv.RETR_TREE,
                cv.CHAIN_APPROX_NONE
            )

            # Draw the comtours in the image
            contour_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

            for cnt in iter(range(len(contours))):
                if (hier[0][cnt][3] < 0):
                    cv.drawContours(contour_image, contours[cnt], -1, (0,0,255), 2)
                    number_cubes[i] += 1
                    if (hier[0][cnt][2] > 0):
                        faulty_cubes[i] += 1

            cv.imshow("Cubes detected", contour_image)
            cv.waitKey(500)

        print(f"The Quantity of objects is:")
        print(f"Yellow: {number_cubes[0]}")
        print(f"Blue: {number_cubes[1]}")
        print(f"Red: {number_cubes[2]}")
        print(f"Green: {number_cubes[3]}")

        print(f"The Quantity of faulty objects is:")
        print(f"Yellow: {faulty_cubes[0]}")
        print(f"Blue: {faulty_cubes[1]}")
        print(f"Red: {faulty_cubes[2]}")
        print(f"Green: {faulty_cubes[3]}")


    def cube_initial_diagnostic(self):
        self.cube_camera_buffer()
        self.get_image_cube_camera()

    def movile_robot_model(self, vx, vy, w):
        """
        This is a method that implements the model of a Holonomic 
        movile robot, this model that maps between the linear and 
        angular velocity of the robot and the speed of the wheels.
        """
        
        # Robot's parameters
        L = 0.38578
        l = 0.0
        R = 0.116835

        right_f_vel = (vx + vy - (L + l)*w)/R
        left_f_vel = (-vx + vy + (L + l)*w)/R
        right_b_vel = (-vx + vy + (L + l)*w)/R
        left_b_vel = (vx + vy - (L + l)*w)/R

        wheel_vel = [
            left_b_vel,
            right_b_vel,
            left_f_vel,
            right_f_vel
        ]

        return wheel_vel

def main():
    conveyor_chall = ConveyorChallenge()
    conveyor_chall.cube_initial_diagnostic()

if __name__ == "__main__":
    sys.exit(main())
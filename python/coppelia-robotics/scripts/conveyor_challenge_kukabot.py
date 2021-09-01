# Built-in imports 
import sys 

# External imports 
import cv2 as cv
import numpy as np

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

    def cube_initial_diagnostic(self):
        pass

def main():
    pass

if __name__ == "__main__":
    sys.exit(main())
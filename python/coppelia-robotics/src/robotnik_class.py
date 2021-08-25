# Built-in imports
import sys 

# External imports 
import numpy as np

# Own imports 
import sim

class RobotnikClass:
    """
    This is a python class which contains all the basic 
    funtionalities of the Robotnik_Summit_XL140701 of the 
    CoppeliaSim Edu software, this python class has methods
    to make a specific velocity to the motors of the robot.
    :params client: This is the client instance to connect with 
    Coppelia server.
    """
    def __init__(self, client):
        # Define client
        self.__client = client
        # Create motors
        self.motorWheel = [
            0,
            1,
            2,
            3
        ]
        self.motorError = [
            False,
            False,
            False,
            False
        ]

        # Define the motor's parameters
        self.motorError[0], self.motorWheel[0] = sim.simxGetObjectHandle(
            self.__client,
            "joint_back_left_wheel",
            sim.simx_opmode_oneshot_wait
        )

        self.motorError[1], self.motorWheel[1] = sim.simxGetObjectHandle(
            self.__client,
            "joint_back_right_wheel",
            sim.simx_opmode_oneshot_wait
        )

        self.motorError[2], self.motorWheel[2] = sim.simxGetObjectHandle(
            self.__client,
            "joint_front_left_wheel",
            sim.simx_opmode_oneshot_wait
        )

        self.motorError[3], self.motorWheel[3] = sim.simxGetObjectHandle(
            self.__client,
            "joint_front_right_wheel",
            sim.simx_opmode_oneshot_wait
        )

        # Define the camera's parameters
        _,self.camera = sim.simxGetObjectHandle(
            self.__client,
            "Vision_sensor",
            sim.simx_opmode_blocking
        )

        _, self.resolution, self.image = sim.simxGetVisionSensorImage(
            self.__client,
            self.camera,
            0,
            sim.simx_opmode_streaming
        )

    def move_motors(self, right_vel = 0, left_vel = 0):
        """
        This is a method which allows to give a specific speed 
        to each of the robot's motors, depending of the motor 
        (right motors and left motors).
        :param right_vel: Angular velocity of the right's motors
        :param left_vel: Angular velocity of the left's motors
        """

        # Configuration of the left's motors parameters
        sim.simxSetJointTargetVelocity(
            self.__client,
            self.motorWheel[0],
            left_vel,
            sim.simx_opmode_oneshot_wait
        )

        sim.simxSetJointTargetVelocity(
            self.__client,
            self.motorWheel[2],
            left_vel,
            sim.simx_opmode_oneshot_wait
        )

        # Configuration of the right's motors parameters
        sim.simxSetJointTargetVelocity(
            self.__client,
            self.motorWheel[1],
            -right_vel,
            sim.simx_opmode_oneshot_wait
        )

        sim.simxSetJointTargetVelocity(
            self.__client,
            self.motorWheel[3],
            -right_vel,
            sim.simx_opmode_oneshot_wait
        )
    
    def camera_buffer(self):
        """
        This is a method which allows to get an image from the 
        visual sensor of the CoppeliaSim Edu software.
        """
        _, self.resolution, self.image = sim.simxGetVisionSensorImage(
            self.__client,
            self.camera,
            0,
            sim.simx_opmode_buffer
        )


def main():
    # End connexion 
    sim.simxFinish(-1)

    # Create new connexion
    clientID = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)

    if (clientID != -1):
        print("Connexion OK")
    else:
        print("Fatal error - No connexion")

    robot = RobotnikClass(clientID)    
    robot.move_motors(-2,2)

    # End connexion 
    sim.simxFinish(-1)  

if __name__ == "__main__":
    sys.exit(main())
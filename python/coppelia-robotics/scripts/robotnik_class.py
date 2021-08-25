# Built-in imports
import sys 

# External imports
import numpy as np

# Own imports 
import sim

class RobotnikClass:
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
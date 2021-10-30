# Built-in imports
import sys 

# Own imports 
import sim

class KukaYouBotClass:
    """
    This is a python class which contains all the basic 
    funtionalities of the KUKA YouBot of the 
    CoppeliaSim Edu software, this python class has methods
    to make a specific velocity to the motors of the robot.
    :params client: This is the client instance to connect with 
    Coppelia server.
    """
    def __init__(self, client):
        # Define client
        self.__client = client
        # Create motors of the movile robot
        self.motorWheel = [
            0,
            1,
            2,
            3
        ]
        # Create joints of the manipulator robot
        self.robot_joints = list()
        self.robot_gripper = list()

        self.define_motors_movile_robot()

        self.define_motors_manipulator_robot()

        self.define_camera_sensor()

    def define_motors_movile_robot(self):
        # Define the movile robot motor's parameters
        _, self.motorWheel[0] = sim.simxGetObjectHandle(
            self.__client,
            "rollingJoint_rl",
            sim.simx_opmode_oneshot_wait
        )

        _, self.motorWheel[1] = sim.simxGetObjectHandle(
            self.__client,
            "rollingJoint_rr",
            sim.simx_opmode_oneshot_wait
        )

        _, self.motorWheel[2] = sim.simxGetObjectHandle(
            self.__client,
            "rollingJoint_fl",
            sim.simx_opmode_oneshot_wait
        )

        _, self.motorWheel[3] = sim.simxGetObjectHandle(
            self.__client,
            "rollingJoint_fr",
            sim.simx_opmode_oneshot_wait
        )
    
    def define_motors_manipulator_robot(self):
        # Define the manipulator robot motor's parameters
        for i in iter(range(5)):
            self.robot_joints.append(sim.simxGetObjectHandle(
                self.__client,
                f"youBotArmJoint{i}",
                sim.simx_opmode_oneshot_wait
            ))

        for i in iter(range(1,3)):
            self.robot_gripper.append(sim.simxGetObjectHandle(
                self.__client,
                f"youBotGripperJoint{i}",
                sim.simx_opmode_oneshot_wait
            ))

    def define_camera_sensor(self):
        # Define the camera's parameters
        self.camera = list()
        self.resolution = list()
        self.image = list()

        # Configure the 4 vision sensors of the robot
        for c in iter(range(4)):
            _,self.camera[c] = sim.simxGetObjectHandle(
                self.__client,
                f"VS{c+1}",
                sim.simx_opmode_blocking
            )

            _, self.resolution[c], self.image[c] = sim.simxGetVisionSensorImage(
                self.__client,
                self.camera[c],
                0,
                sim.simx_opmode_streaming
            )

    def camera_buffer(self):
        """
        This is a method which allows to get an image from the 
        visual sensors of the CoppeliaSim Edu software.
        """
        for c in iter(range(4)):
            _, self.resolution[c], self.image[c] = sim.simxGetVisionSensorImage(
                self.__client,
                self.camera[c],
                0,
                sim.simx_opmode_buffer
            )

    def move_movile_robot_motors(self, wheels_velocity = [0,0,0,0]):
        """
        This is a method which allows to give a specific speed 
        to each of the robot's motors, depending of the motor 
        (right motors and left motors).
        :param wheels_velocity which is a list with 4 values of 
        velocity.
            wheels_velocity = [
                vel_left_r,
                vel_right_r,
                vel_left_f,
                vel_right_f
            ]
        """

        # Configuration of the left's motors parameters
        sim.simxSetJointTargetVelocity(
            self.__client,
            self.motorWheel[0],
            -wheels_velocity[0],
            sim.simx_opmode_oneshot_wait
        )

        sim.simxSetJointTargetVelocity(
            self.__client,
            self.motorWheel[2],
            -wheels_velocity[2],
            sim.simx_opmode_oneshot_wait
        )

        # Configuration of the right's motors parameters
        sim.simxSetJointTargetVelocity(
            self.__client,
            self.motorWheel[1],
            -wheels_velocity[1],
            sim.simx_opmode_oneshot_wait
        )

        sim.simxSetJointTargetVelocity(
            self.__client,
            self.motorWheel[3],
            -wheels_velocity[3],
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

    robot = KukaYouBotClass(clientID)    
    robot.move_movile_robot_motors([5,5,5,5])

    # End connexion 
    sim.simxFinish(-1)  

if __name__ == "__main__":
    sys.exit(main())
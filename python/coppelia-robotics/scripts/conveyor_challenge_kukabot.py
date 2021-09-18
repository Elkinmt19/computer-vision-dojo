# Built-in imports 
import sys 
import time

# External imports 
import cv2 as cv
import numpy as np

# Own imports 
import sim
import kuka_youbot_class as kb

class ConveyorChallenge:
    """
    This is a python class which contains methods that are important 
    to solve the conveyor challenge of the computer vision class of
    the EIA University.
    """
    def __init__(self):
        # Define camera's parameters
        self.cube_camera = [None, None]
        self.image = [None, None]
        self.resolution = [None, None]
        self.cameras_name = [
            "Vision_sensor_cubes",
            "Vision_sensor_package"
        ]

        # Define parameters of the conexion 
        self.start_conextion()
        self.define_cube_camera()

        # Define robot's parameters
        self.ready_to_go = False
        self.safety_counter = 0
        self.robot_target = 0
        self.change_target = 0
        self.robot = kb.KukaYouBotClass(self.__clientID)

        # Define cubes and floor parameters
        self.cubes = [None, None, None, None, None]
        self.floor = None
        self.define_cubes_floor()

        # Control variables
        self.__control_period = 0.1
        self.__error = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.__amount_i_term = [0.0,0.0,0.0]
        self.__orientation_setpoint = 0.0
        self.__distance_setpoint = 0.09

        # Measurement variables
        self.area = 0.0
        self.cx = 0
        self.distance = 0


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
        for i in iter(range(2)):
            _,self.cube_camera[i] = sim.simxGetObjectHandle(
                self.__clientID,
                self.cameras_name[i],
                sim.simx_opmode_blocking
            )

            _, self.resolution[i], self.image[i] = sim.simxGetVisionSensorImage(
                self.__clientID,
                self.cube_camera[i],
                0,
                sim.simx_opmode_streaming
            )

    def cube_camera_buffer(self, cam):
        """
        This is a method which allows to get an image from the 
        visual sensor of the CoppeliaSim Edu software.
        """
        _, self.resolution[cam], self.image[cam] = sim.simxGetVisionSensorImage(
            self.__clientID,
            self.cube_camera[cam],
            0,
            sim.simx_opmode_buffer
        )

    def define_cubes_floor(self):
        cube_names = ["Cuboid9", "Cuboid4", "Cuboid6", "Cuboid7", "Cuboid5"]
        for i in iter(range(5)):
            _, self.cubes[i] = sim.simxGetObjectHandle(
                self.__clientID,
                cube_names[i],
                sim.simx_opmode_oneshot_wait
            )
        _, self.floor = sim.simxGetObjectHandle(
                self.__clientID,
                "Floor",
                sim.simx_opmode_oneshot_wait
            )

    def move_cube(self, cube):
        position = np.array(
            [
                [-0.2250,-1.3250,0.0925],
                [-0.2250,-0.8250,0.0925],
                [-0.2250,-1.3250,0.0925],
                [-0.2250,0.3000,0.0925],
                [-0.2250,-0.2750,0.0925],
            ]
        )
        sim.simxSetObjectPosition(
            self.__clientID,
            self.cubes[cube],
            self.floor,
            position[cube],
            sim.simx_opmode_oneshot_wait
    )

    def move_cubes_to_conveyor(self):
        origin_conveyor = [-1.0000e+00,7.5000e-01,2.7500e-01]
        for i in iter(range(5)):
            sim.simxSetObjectPosition(
                self.__clientID,
                self.cubes[i],
                self.floor,
                origin_conveyor,
                sim.simx_opmode_oneshot_wait
            )
            time.sleep(4)

    def get_image_cube_camera(self):
        """
        This is a method which counts the number of cubes in the table, 
        detects and segments the objects according their color and also 
        verify the quality of these objects. 
        """
        number_cubes = [0,0,0,0]
        faulty_cubes = [0,0,0,0]

        # Preprocessing of the image
        img = np.array(self.image[0], dtype = np.uint8)
        img.resize([self.resolution[0][0], self.resolution[0][1], 3])
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
        cv.destroyAllWindows()

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
        self.cube_camera_buffer(0)
        self.get_image_cube_camera()

    def get_image_cube_package(self):
        """
        This is a method which tells the movile robot when it can 
        start its way to bring the objects to their places.
        """
        # Preprocessing of the image
        img = np.array(self.image[1], dtype = np.uint8)
        img.resize([self.resolution[1][0], self.resolution[1][1], 3])
        img = np.rot90(img,2)
        img = np.fliplr(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        # Umbral's values for the segmentation of the objects
        LOWER_VALUES = np.array([0, 130, 0])
        UPPER_VALUES = np.array([62, 255, 131])

        binary_image = cv.inRange(img, LOWER_VALUES, UPPER_VALUES)

        kernelD = np.ones((5,9), np.uint8)
        erode_image = cv.erode(binary_image, kernelD, iterations=2)

        # Find the contours of the image
        contours, hier = cv.findContours(
            erode_image.copy(),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE
        )

        if ((len(contours) > 0) and (self.safety_counter >= 75)):
            self.ready_to_go = True

        self.safety_counter += 1

    def cube_package_diagnostic(self):
        self.cube_camera_buffer(1)
        self.get_image_cube_package()

    def movile_robot_model(self, vx, vy, w):
        """
        This is a method that implements the model of a Holonomic 
        movile robot, this model that maps between the linear and 
        angular velocity of the robot and the speed of the wheels.
        """
        
        # Robot's parameters
        L = 0.23506600000000002
        l = 0.104311
        R = 0.049984

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

    def get_current_orientation_distance(self):
        """
        This is a method which get the distance of the robot with respect 
        an object using image-processing and computer vision, this method 
        also get the relative orientation of the robot with respect to the 
        centroid of an external object.
        """
        # Preprocessing of the image 
        img = np.array(self.robot.image, dtype = np.uint8)
        img.resize([self.robot.resolution[0], self.robot.resolution[1], 3])
        img = np.rot90(img,2)
        img = np.fliplr(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # Umbral's values for the segmentation of the objects
        LOWER_VALUES = np.array(
            [
                [83, 0, 55],
                [39, 164, 92],
                [23, 64, 68],
                [0, 88, 88]
            ]
        )
        UPPER_VALUES = np.array(
            [
                [136, 255, 255],
                [78, 255, 209],
                [52, 255, 207],
                [2, 255, 250]
            ]
        )

        if (self.change_target >= 5 and self.robot_target < 3):
            self.robot_target += 1
            self.change_target = 0

        binary_image = cv.inRange(
            hsv_image,
            LOWER_VALUES[self.robot_target],
            UPPER_VALUES[self.robot_target]
        )

        # Find the contours of the image
        contours, hier = cv.findContours(
            binary_image.copy(),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE
        )

        # Draw the comtours in the image
        contour_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

        CONST_INV = 240

        for cnt in contours:
            self.area = cv.contourArea(cnt)

            if (self.area > 0):
                M = cv.moments(cnt)
                self.cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                cv.drawContours(contour_image, cnt, -1, (255,0,0), 2)
                cv.circle(contour_image, (self.cx,cy), 1, (0,255,0), 2)
        try:
            self.distance = CONST_INV/self.area
        except:
            self.distance = self.__distance_setpoint
        self.cx = (2/510)*self.cx - 1

        cv.imshow("Binary Image", contour_image)

        return self.cx, self.distance

    def orientation_pid_controller(self):
        """
        This method implements a simple PID controller in order 
        to control the orientation of the robot in each moment of 
        the simulation.
        """

        # Controller's parameters
        KP = 0.4
        KI = 0.0
        KD = 0.001

        p_term = self.__error[1]
        i_term = self.__error[1]*self.__control_period + self.__amount_i_term[0]
        d_term = (self.__error[1] - self.__error[0])/self.__control_period

        vx = KP*p_term + KI*i_term + KD*d_term

        self.__amount_i_term[0] += i_term
        self.__error[0] = self.__error[1]

        return vx

    def distance_pid_controller(self):
        """
        This method implements a simple PID controller in order 
        to control the distance between the robot and an external 
        object in each moment of the simulation.
        """

        # Controller's parameters
        KP = -0.8
        KI = -0.0
        KD = -0.03

        p_term = self.__error[3]
        i_term = self.__error[3]*self.__control_period + self.__amount_i_term[1]
        d_term = (self.__error[3] - self.__error[2])/self.__control_period

        vy = KP*p_term + KI*i_term + KD*d_term

        self.__amount_i_term[1] += i_term
        self.__error[2] = self.__error[3]

        return vy

    def execute_control(self):
        self.robot = kb.KukaYouBotClass(self.__clientID)
        last_time = 0.0
        vx = 0.0
        vy = 0.0
        # Only proceed to control calculation in correct sample time multiple
        sample_time_condition = time.time() - last_time >= self.__control_period
        # Camera condiction 
        camera_condition = len(self.robot.resolution) > 1
        ready_to_go_stop = True
        self.cube_initial_diagnostic()
        self.move_cubes_to_conveyor()
        while (True):
            if (ready_to_go_stop):
                self.cube_package_diagnostic()
            if (self.ready_to_go):
                if (ready_to_go_stop):
                    self.safety_counter = 0
                    time.sleep(5)
                    print("READY TO GO!!")
                    ready_to_go_stop = False
                self.robot.camera_buffer()
                if (sample_time_condition and camera_condition):
                    current_orientation, current_distance = self.get_current_orientation_distance()

                    self.__error[1] = self.__orientation_setpoint - current_orientation
                    self.__error[3] = self.__distance_setpoint - current_distance

                    if (abs(self.__error[1]) > 0.05):
                        vx = self.orientation_pid_controller()
                    else:
                        vx = 0
                    if (abs(self.__error[3]) > 0.01 and (self.robot_target == 0)):
                        vy = self.distance_pid_controller()
                    else:
                        vy = 0
                    wheel_speed = self.movile_robot_model(vx, vy, 0)
                    self.robot.move_movile_robot_motors(wheel_speed)

                    if ((vx == 0) and (vy == 0)):
                        self.change_target +=  1

                    # Move the cubes
                    if (self.change_target == 3):
                        if (self.robot_target == 0):
                            self.move_cube(3)
                        if (self.robot_target == 1):
                            self.move_cube(4)
                        if (self.robot_target == 2):
                            self.move_cube(1)
                        if (self.robot_target == 3):
                            self.move_cube(0)
                            self.move_cube(2)

                    last_time = time.time()
                if ((cv.waitKey(1) & 0xFF == ord('q')) or ((self.change_target >= 5) and (self.robot_target == 3))):
                    print("We are MELOS!!!")
                    break
        
        self.stop_conextion()

def main():
    conveyor_chall = ConveyorChallenge()
    conveyor_chall.execute_control()

if __name__ == "__main__":
    sys.exit(main())
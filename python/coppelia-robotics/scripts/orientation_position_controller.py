# Buit-in imports 
import sys
import time 

# External imports 
import cv2 as cv
import numpy as np

# Own imports 
import sim
import robotnik_class as rn


class OrientationPositionController:
    """
    This is a python class which makes a model of a movile
    robot and controls it using that model, this class
    has methods that allows to control the orientation an 
    position of a movile robot using a simple PID controller,
    using a camara as the sensor and the robot's motors as the 
    actuators of the system.
    """
    def __init__(self, control_period):
        # Control variables
        self.__control_period = control_period
        self.__error = [0.0,0.0,0.0,0.0]
        self.__amount_i_term = [0.0,0.0]
        self.__orientation_setpoint = 0.0
        self.__distance_setpoint = 0.5

        # Measurement variables
        self.area = 0.0
        self.cx = 0
        self.distance = 0


    def robot_model(self, v, w):
        """
        This is a method that implements the model of a differential 
        movile robot, this model that maps between the linear and 
        angular velocity of the robot and the speed of the wheels.
        """
        
        # Robot's parameters
        L = 0.38578
        R = 0.116835

        right_vel = (1/R)*v + (L/2*R)*w
        left_vel = (1/R)*v - (L/2*R)*w

        return right_vel, left_vel

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

        # Segmentation of the object
        hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        lower_values = np.array([83, 0, 55])
        upper_values = np.array([136, 255, 255])
        binary_image = cv.inRange(hsv_image, lower_values, upper_values)

        # Find the contours of the image
        contours, hier = cv.findContours(
            binary_image.copy(),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE
        )

        # Draw the comtours in the image
        h,w = img.shape[:2]
        contour_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

        CONST_INV = 0.5265*54289.0

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
        cv.imshow("Original Image", img)

        return self.cx, self.distance

    def orientation_pid_controller(self):
        """
        This method implements a simple PID controller in order 
        to control the orientation of the robot in each moment of 
        the simulation.
        """

        # Controller's parameters
        KP = 50.0
        KI = 0.0
        KD = 5.0

        p_term = self.__error[1]
        i_term = self.__error[1]*self.__control_period + self.__amount_i_term[0]
        d_term = (self.__error[1] - self.__error[0])/self.__control_period

        w = KP*p_term + KI*i_term + KD*d_term

        self.__amount_i_term[0] += i_term
        self.__error[0] = self.__error[1]

        return w

    def distance_pid_controller(self):
        """
        This method implements a simple PID controller in order 
        to control the distance between the robot and an external 
        object in each moment of the simulation.
        """

        # Controller's parameters
        KP = -0.3
        KI = 0.0
        KD = -0.005

        p_term = self.__error[3]
        i_term = self.__error[3]*self.__control_period + self.__amount_i_term[1]
        d_term = (self.__error[3] - self.__error[2])/self.__control_period

        v = KP*p_term + KI*i_term + KD*d_term

        self.__amount_i_term[1] += i_term
        self.__error[2] = self.__error[3]

        return v


    def execute_control(self):
        """
        This is a method that executes a orientation and position 
        control of a movile robot.
        """
        # End connexion 
        sim.simxFinish(-1)

        # Create new connexion
        clientID = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)

        if (clientID != -1):
            print("Connexion OK")
        else:
            print("Fatal error - No connexion")

        self.robot = rn.RobotnikClass(clientID)

        last_time = 0.0
        w = 0
        v = 0    

        while (True):
            # Only proceed to control calculation in correct sample time multiple
            sample_time_condition = time.time() - last_time >= self.__control_period
            # Camera condiction 
            camera_condition = len(self.robot.resolution) > 1

            self.robot.camera_buffer()

            # Control loop 
            if (sample_time_condition and camera_condition):
                current_orientation, current_distance = self.get_current_orientation_distance()

                self.__error[1] = self.__orientation_setpoint - current_orientation
                self.__error[3] = self.__distance_setpoint - current_distance

                w = self.orientation_pid_controller()
                v = self.distance_pid_controller()
                right_vel, left_vel = self.robot_model(v,w)

                self.robot.move_motors(right_vel, left_vel)

            if (cv.waitKey(1) & 0xFF == ord('q')):
                break
        
        # End connexion 
        sim.simxFinish(-1)

def main():
    or_controller = OrientationPositionController(0.1)
    or_controller.execute_control()

if __name__ == "__main__":
    sys.exit(main())
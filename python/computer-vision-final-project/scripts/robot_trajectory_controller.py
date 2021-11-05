# Built-in imports 
import sys 
import time

# Own imports 
import sim
import kuka_youbot_autopilot_class as kb

class TrajectoryController:
    def __init__(self, client_id):
        self.__clientID = client_id

        _,self.robot_entity = sim.simxGetObjectHandle(
            self.__clientID,
            'youBot#2',
            sim.simx_opmode_oneshot_wait
        )
        _,self.floor = sim.simxGetObjectHandle(
            self.__clientID,
            'Floor',
            sim.simx_opmode_oneshot_wait
        )

        self.robot = kb.KukaYouBotClass(self.__clientID)
    
    def mobile_robot_model(self, vx, vy, w):
        """
        This is a method that implements the model of a Holonomic 
        mobile robot, this model that maps between the linear and 
        angular velocity of the robot and the speed of the wheels.
        """
        
        # Robot's parameters
        L = 0.23506600000000002
        l = 0.104311
        R = 0.049984

        right_f_vel = (-vx + vy + (L + l)*w)/R
        left_f_vel = (vx + vy - (L + l)*w)/R
        right_b_vel = (vx + vy + (L + l)*w)/R
        left_b_vel = (-vx + vy - (L + l)*w)/R

        wheel_vel = [
            left_b_vel,
            right_b_vel,
            left_f_vel,
            right_f_vel
        ]

        return wheel_vel

    def get_position(self):
        """
        This a simple python method that allows to get the current 
        position ìn the plane [x,y] of a mobile robot.
        """
        #Get robot's position
        position = (sim.simxGetObjectPosition(
            self.__clientID,
            self.robot_entity,
            self.floor,
            sim.simx_opmode_oneshot_wait
        ))

        return (position[1][0],position[1][1])

    def get_orientation(self): 
        """
        This a simple python method that allows to get the current 
        orientation with respect to the z axis of a mobile robot.
        """   
        #Get robot's orientation
        orientation = (sim.simxGetObjectOrientation(
            self.__clientID,
            self.robot_entity,
            self.floor,
            sim.simx_opmode_oneshot_wait
        ))

        return orientation[1][1]

    def go_to_angle(self, desired_angle, error_k_1, ia_acum, ts):
        """
        This is a python method which implements a simple PID controller
        in order to control the orientation of a mobile robot using 
        its pose model and the angular velocity as control signal.
        :param: desired_angle: Desired angle to reach by the robot
        :param: error_k_1: First delay of the error signal
        :param: ia_acum: Amount of area of the integral term 
        :param: ts: Control period of time in seconds
        """
        # Define the controller's parameters
        KP = 1.2
        KI = 0.0
        KD = 0.0    
        # Calculate the error signal 
        error  = -desired_angle + self.get_orientation()

        # Calculate the control signal 
        p_term = error
        i_term = error*ts + ia_acum
        d_term = (error - error_k_1)/ts
        w = p_term*KP + i_term*KI + d_term*KD

        return w, error, i_term


    def go_to_goal(self, desired_position, error_k_1, ia_acum, ts):
        """
        This is a python method that implements a simple PID controller 
        in order to control the location in the space (x position and y position)
        of a mobile robot using the pose model of the robot and the speed in 
        each direction as control signals.
        :param: desired_angle: Desired angle to reach by the robot
        :param: error_k_1: First delay of the error signal
        :param: ia_acum: Amount of area of the integral term 
        :param: ts: Control period of time in seconds
        """
        # Define the controller's parameters
        KP = [0.7,0.7]
        KI = [0,0]
        KD = [0,0]

        error = [0,0]
        i_term = [0,0]
        v = [0,0]

        for i in iter(range(2)):
            # Calculate the error signal
            position = self.get_position()
            error[i]  = desired_position[i] - position[i]

            # Calculate the control signal 
            p_term = error[i]
            i_term[i] = error[i]*ts + ia_acum[i]
            d_term = (error[i] - error_k_1[i])/ts
            v[i] = p_term*KP[i] + i_term[i]*KI[i] + d_term*KD[i]

        return v, error, i_term

    def avoid_obstacles(self):
        pass

    def monitoring_variables(self):
        while (True):
            position = self.get_position()
            orientation = self.get_orientation()

            print(f"Position: {position} orientation: {orientation}")

def test_go_to_angle(clientID):
    controller = TrajectoryController(clientID)

    # Control variables
    control_period = 0.1
    error = [0.0,0.0]
    it_acum = 0
    it_term_k_1 = 0
    setpoint = -1.5
    eps = 0.001

    last_time = 0
    w = 0

    # Only proceed to control calculation in correct sample time multiple
    sample_time_condition = time.time() - last_time >= control_period

    while (1):
        if (sample_time_condition):
            w, error[0], it_term_k_1 = controller.go_to_angle(setpoint,error[0],it_acum,control_period)

            wheel_speed = controller.mobile_robot_model(0,0,w)

            if (abs(error[0]) >= 0 - eps and abs(error[0]) <= 0 + eps):
                controller.robot.move_mobile_robot_motors([0,0,0,0])
            else:
                controller.robot.move_mobile_robot_motors(wheel_speed)                

            it_acum += it_term_k_1


def test_go_to_goal(clientID):
    controller = TrajectoryController(clientID)

    # Control variables
    control_period = 0.1
    error = [0.0,0.0,0.0,0.0]
    it_acum = [0.0,0.0]
    it_term_k_1 = [0.0,0.0]
    setpoint = [11.22,-9.82]
    eps = 0.001

    last_time = 0
    v = [0,0]

    # Only proceed to control calculation in correct sample time multiple
    sample_time_condition = time.time() - last_time >= control_period

    while (1):
        if (sample_time_condition):
            v, buff_error, it_term_k_1 = controller.go_to_goal(setpoint,[error[0],error[2]],it_acum,control_period)
            error[0], error[2] = (buff_error[0], buff_error[1])

            wheel_speed = controller.mobile_robot_model(v[0],v[1],0)

            # Control condition action for x position control
            x_condition = (abs(error[0]) >= 0 - eps and abs(error[0]) <= 0 + eps)
            # Control condition action for x position control
            y_condition = (abs(error[2]) >= 0 - eps and abs(error[2]) <= 0 + eps)

            if (x_condition and y_condition):
                controller.robot.move_mobile_robot_motors([0,0,0,0])
            else:
                controller.robot.move_mobile_robot_motors(wheel_speed) 

            it_acum[0] += it_term_k_1[0]           
            it_acum[1] += it_term_k_1[1]



def main():
    # End connection 
    sim.simxFinish(-1)

    # Create new connection
    clientID = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)

    if (clientID != -1):
        print("Connection OK")
    else:
        print("Fatal error - No connection")

    # Test function
    # test_go_to_goal(clientID)

    # End connection 
    sim.simxFinish(-1)  

if __name__ == "__main__":
    sys.exit(main())
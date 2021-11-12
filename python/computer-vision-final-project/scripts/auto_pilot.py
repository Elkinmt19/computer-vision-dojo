# Built-in imports 
import sys 
import time

# External imports
import numpy as np

# Own imports 
import sim
import robot_trajectory_controller as rc
import avoid_obstacles_dl as ao

class AutoPilot:
    def __init__(self):
        # Start the connection with CoppeliaSim
        self.start_connection()

        # Define control variables
        self.controller = rc.TrajectoryController(self.__clientID)
        self.control_period = 0.1
        self.error = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.it_acum = [0.0,0.0,0.0]
        self.setpoint = [11.5,1.5,0.0] 
        self.eps = 0.05

        # Trajectory pointer
        self.pointer = 0
        self.coor_count = 0

        # Define trajectory coordinates
        self.trajectory_coordinates = [
            (11.5,1.5),
            (1,0.8),
            (1,-11),
            (4.0124,-11),
            (4.0124,-3.9314)
        ]

        self.trajectory_orientation = [-np.pi/2,0.0,np.pi/2,0.0,0.0]

    def start_connection(self):
        # End connection 
        sim.simxFinish(-1)

        # Create new connection
        self.__clientID = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)

        if (self.__clientID != -1):
            print("connection OK")
        else:
            print("Fatal error - No connection")

    def stop_connection(self):
        # End connection 
        sim.simxFinish(-1)

    def avoid_obstacles_controller(self):
        self.controller.robot.camera_buffer()
        self.avoid_obs_controller = ao.AvoidObstaclesDL(
            self.controller.robot,
            True)
        for cm in self.avoid_obs_controller.cameras_threads:
            cm.start()

        # for cm in self.avoid_obs_controller.cameras_threads:
        #     cm.join()

    def execute_autopilot(self):
        # Define delay for the control loop
        last_time = 0

        while (1):
            # t0 = time.time()
            # Execute the avoid obstacle method
            self.avoid_obstacles_controller()

            # Update setpoint values
            self.setpoint[0:2] = [
                self.trajectory_coordinates[self.coor_count][0],
                self.trajectory_coordinates[self.coor_count][1]
            ]

            self.setpoint[2] = self.trajectory_orientation[self.coor_count]

            # Only proceed to control calculation in correct sample time multiple
            sample_time_condition = time.time() - last_time >= self.control_period
            if (sample_time_condition):
                # Go to goal controller (X, Y)
                v, buff_error, it_term_k_1 = self.controller.go_to_goal(
                    self.setpoint[0:2],
                    [self.error[0],self.error[2]],
                    self.it_acum[0:2],
                    self.control_period
                )

                # Go to angle controller (Theta)
                w, buff_error_angle, it_term_k_1_angle = self.controller.go_to_angle(
                    self.setpoint[2],
                    self.error[4],
                    self.it_acum[2],
                    self.control_period
                ) 

                self.error[0], self.error[2], self.error[4] = (buff_error[0], buff_error[1], buff_error_angle)

                if (self.pointer == 0):
                    wheel_speed = self.controller.mobile_robot_model(v[0],v[1],0)
                elif (self.pointer == 1):
                    wheel_speed = self.controller.mobile_robot_model(v[1],-v[0],0)
                elif (self.pointer == 2):
                    wheel_speed = self.controller.mobile_robot_model(-v[0],-v[1],0)
                elif (self.pointer == 3):
                    wheel_speed = self.controller.mobile_robot_model(-v[1],v[0],0)

                # Control condition action for x position control
                x_condition = (abs(self.error[0]) >= 0 - self.eps and abs(self.error[0]) <= 0 + self.eps)
                # Control condition action for x position control
                y_condition = (abs(self.error[2]) >= 0 - self.eps and abs(self.error[2]) <= 0 + self.eps)
                # Control condition action for angle position
                angle_condition = (abs(self.error[4]) >= 0 - self.eps and abs(self.error[4]) <= 0 + self.eps)

                if (x_condition and y_condition):
                    # print(f"Point arrived!!!")
                    if ((self.pointer == 1) or (self.pointer == 2)):
                        wheel_speed = self.controller.mobile_robot_model(0,0,-w)
                    else:
                        wheel_speed = self.controller.mobile_robot_model(0,0,w)

                    self.controller.robot.move_mobile_robot_motors(wheel_speed)
                    if (angle_condition):
                        # print(f"Moving on!!! to {self.pointer + 1}")
                        self.pointer += 1
                        self.coor_count += 1
                        self.controller.robot.move_mobile_robot_motors([0,0,0,0])
                else:
                    # print("moral")
                    self.controller.robot.move_mobile_robot_motors(wheel_speed)

                # Special validation
                if (self.pointer >= 4):
                    self.pointer = 0

                # End condition
                if (self.coor_count >= len(self.trajectory_coordinates)):
                    break

                # Validation of the avoid obstacles algorithm
                self.controller.avoid_obstacles(self.avoid_obs_controller.cameras)

                print(self.avoid_obs_controller.cameras)

                self.it_acum[0] += it_term_k_1[0]           
                self.it_acum[1] += it_term_k_1[1]
                self.it_acum[2] += it_term_k_1_angle
                last_time = time.time()
            # print(time.time()-t0)

        self.stop_connection()



def main():
    _ = AutoPilot().execute_autopilot()

if __name__ == "__main__":
    sys.exit(main())
# Built-in imports 
import sys 
import time

# External imports
import numpy as np
import cv2 as cv

# Own imports 
import sim
import robot_trajectory_controller as rc
import avoid_obstacles_dl as ao
import control_joints_robotic_arm as rba

class AutoPilot:
    def __init__(self, show_images):
        # Start the connection with CoppeliaSim
        self.start_connection()

        # Flag to show the cameras images
        self.show_images = show_images

        # Define control variables
        self.controller = rc.TrajectoryController(self.__clientID)
        self.control_period = 0.1
        self.error = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.it_acum = [0.0,0.0,0.0]
        self.setpoint = [11.5,1.5,0.0] 
        self.eps = 0.08
        
        # Trajectory pointer
        self.pointer = 0
        self.coor_count = 0

        # Define trajectory coordinates
        self.trajectory_coordinates = [
            (11.77,-8.7225),
            (11.3,-8.7225),
            (11.5,1.5),
            (4.6482,1.5),
            (4.6482,0.6738),
            (1,0.8),
            (1,-11),
            (3.92,-11),
            (8.28,-5.9002),
            (8.28,-7.3502),
            (8.28,-8.6752)
        ]

        self.trajectory_orientation = [
            0.0,
            0.0,
            -np.pi/2,
            -np.pi/2,
            -np.pi/2,
            0.0,
            np.pi/2,
            np.pi/2,
            np.pi/2,
            np.pi/2,
            np.pi/2
        ]
    
    
    def move_fucking_cube(self, fucking_object, v_x, v_y):
        sim.simxSetObjectPosition(self.__clientID,fucking_object,self.controller.robot_entity,[0.078,v_x,v_y],sim.simx_opmode_oneshot_wait)
        
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
            False
        )
        for cm in self.avoid_obs_controller.cameras_threads:
            cm.start()

        # for cm in self.avoid_obs_controller.cameras_threads:
        #     cm.join()

    def set_robot_orientation(self,target_angle, last_angle):
        forward_pointer=0 #default
        if((target_angle==0 and last_angle == np.pi/2) or (last_angle == 0 and target_angle == 0)):
            forward_pointer=0
        elif(target_angle == -np.pi/2 or (last_angle == -np.pi/2  and target_angle == -np.pi/2 )):
            forward_pointer=1
        elif((target_angle==0 and last_angle == -np.pi/2) or (last_angle == 0  and target_angle == 0 )):
            forward_pointer=2
        elif(target_angle == np.pi/2 or (last_angle == np.pi/2  and target_angle == np.pi/2 )):
            forward_pointer=3
        return forward_pointer

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
            
            if(self.coor_count <= 8):
                
                self.move_fucking_cube(rba.cubes[0],-0.004, -0.101)
                self.move_fucking_cube(rba.cubes[1],-0.003, -0.151)
                self.move_fucking_cube(rba.cubes[2],-0.004, -0.201)
            
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
                    print(f"Point arrived!!!")
                                            
                    if (self.coor_count == 8):
                        self.controller.robot.move_mobile_robot_motors([0,0,0,0])
                        rba.move_blue_block()
                    elif (self.coor_count == 9):
                        self.controller.robot.move_mobile_robot_motors([0,0,0,0])
                        rba.move_red_block()
                    elif (self.coor_count == 10):
                        self.controller.robot.move_mobile_robot_motors([0,0,0,0])
                        rba.move_yellow_block()
                        
                    if ((self.coor_count in [5, 6, 7])):
                        wheel_speed = self.controller.mobile_robot_model(0,0,-w)
                    else:
                        wheel_speed = self.controller.mobile_robot_model(0,0,w)

                    self.controller.robot.move_mobile_robot_motors(wheel_speed)
                    if (angle_condition):
                        print(f"Moving on!!! to {self.coor_count + 1}")
                        if (self.coor_count == 0):
                            last_angle = 0
                        else:
                            last_angle = self.trajectory_orientation[self.coor_count-1]

                        self.pointer = self.set_robot_orientation(
                            self.trajectory_orientation[self.coor_count],
                            last_angle
                        )
                        self.coor_count += 1
                        self.controller.robot.move_mobile_robot_motors([0,0,0,0])
                else:
                    print("Going!!!")
                    self.controller.robot.move_mobile_robot_motors(wheel_speed)

                # End condition
                if (self.coor_count >= len(self.trajectory_coordinates)):
                    self.controller.robot.move_mobile_robot_motors([0,0,0,0])
                    break
   
                # Validation of the avoid obstacles algorithm
                # self.controller.avoid_obstacles(self.avoid_obs_controller.cameras)

                self.it_acum[0] += it_term_k_1[0]           
                self.it_acum[1] += it_term_k_1[1]
                self.it_acum[2] += it_term_k_1_angle
                last_time = time.time()

                # Show cameras images
                if (self.show_images):
                    for idx in self.avoid_obs_controller.index_camera:
                        cv.imshow(
                            f"Camera {idx}",
                            self.avoid_obs_controller.camera_image[idx]
                        )
                        cv.waitKey(1)

            # print(time.time()-t0)

        self.stop_connection()



def main():
    _ = AutoPilot(True).execute_autopilot()

if __name__ == "__main__":
    sys.exit(main())
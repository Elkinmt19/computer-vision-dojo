# Built-in imports 
import sys 
import time

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
        self.error = [0.0,0.0,0.0,0.0]
        self.it_acum = [0.0,0.0]
        self.setpoint = [11.775,-4.8239] 
        self.eps = 0.001
        self.v = [0,0]
        self.w = 0

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
            False)
        for cm in self.avoid_obs_controller.cameras_threads:
            cm.start()

        # for cm in self.avoid_obs_controller.cameras_threads:
        #     cm.join()

    def execute_autopilot(self):
        # Define delay for the control loop
        last_time = 0

        while (1):
            t0 = time.time()
            # Execute the avoid obstacle method
            self.avoid_obstacles_controller()

            # Only proceed to control calculation in correct sample time multiple
            sample_time_condition = time.time() - last_time >= self.control_period
            if (sample_time_condition):
                v, buff_error, it_term_k_1 = self.controller.go_to_goal(
                    self.setpoint,
                    [self.error[0],self.error[2]],
                    self.it_acum,
                    self.control_period
                )
                self.error[0], self.error[2] = (buff_error[0], buff_error[1])

                wheel_speed = self.controller.mobile_robot_model(v[0],v[1],0)

                # Control condition action for x position control
                x_condition = (abs(self.error[0]) >= 0 - self.eps and abs(self.error[0]) <= 0 + self.eps)
                # Control condition action for x position control
                y_condition = (abs(self.error[2]) >= 0 - self.eps and abs(self.error[2]) <= 0 + self.eps)

                if (x_condition and y_condition):
                    self.controller.robot.move_mobile_robot_motors([0,0,0,0])
                else:
                    self.controller.robot.move_mobile_robot_motors(wheel_speed)

                # Validation of the avoid obstacles algorithm
                self.controller.avoid_obstacles(self.avoid_obs_controller.cameras)

                print(self.avoid_obs_controller.cameras)

                self.it_acum[0] += it_term_k_1[0]           
                self.it_acum[1] += it_term_k_1[1]
                last_time = time.time()
            print(time.time()-t0)

        self.start_connection()



def main():
    _ = AutoPilot().execute_autopilot()

if __name__ == "__main__":
    sys.exit(main())
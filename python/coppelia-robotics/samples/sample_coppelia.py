# External imports 
import sim 

# End connexion 
sim.simxFinish(-1)

# Create new connexion
clientID = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)

if (clientID != -1):
    print("Connexion OK")
else:
    print("Fatal error - No connexion")

# Create motors
motorWheel = [0, 1, 2, 3]
motorError = [False, False, False, False]

motorError[0], motorWheel[0] = sim.simxGetObjectHandle(
    clientID,
    "joint_back_left_wheel",
    sim.simx_opmode_oneshot_wait
)

motorError[1], motorWheel[1] = sim.simxGetObjectHandle(
    clientID,
    "joint_back_right_wheel",
    sim.simx_opmode_oneshot_wait
)

# Define the speed
speed = 0.1

# Move the motors
sim.simxSetJointTargetVelocity(
    clientID,
    motorWheel[0],
    speed,
    sim.simx_opmode_oneshot_wait
)

sim.simxSetJointTargetVelocity(
    clientID,
    motorWheel[1],
    speed,
    sim.simx_opmode_oneshot_wait
)


# End connexion 
sim.simxFinish(-1)

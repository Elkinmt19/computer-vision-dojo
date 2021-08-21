# External imports 
import numpy as np
import cv2 as cv

# Own imports 
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

err_code,camera = sim.simxGetObjectHandle(
    clientID,
    "Vision_sensor",
    sim.simx_opmode_blocking
)

returnCode, resolution, image = sim.simxGetVisionSensorImage(
    clientID,
    camera,
    0,
    sim.simx_opmode_streaming
)

# Define the speed
speed = 0.5

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

while (True):
    _, resolution, image = sim.simxGetVisionSensorImage(
        clientID,
        camera,
        0,
        sim.simx_opmode_buffer
    )
    print(resolution)
    if (len(resolution) > 1):
        img = np.array(image, dtype = np.uint8)
        img.resize([resolution[0], resolution[1], 3])
        img = np.rot90(img,2)
        img = np.fliplr(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imshow("ImgCamera", img)
    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break


# End connexion 
sim.simxFinish(-1)

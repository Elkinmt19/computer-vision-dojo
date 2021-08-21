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
speed = 0.2

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
    -speed,
    sim.simx_opmode_oneshot_wait
)

def getting_started_filters():
    while (True):
        _, resolution, image = sim.simxGetVisionSensorImage(
            clientID,
            camera,
            0,
            sim.simx_opmode_buffer
        )

        kernelD = np.ones((5,9), np.uint8)
        kernelA = cv.getStructuringElement(cv.MORPH_RECT, (9,9))
        kernelB = np.array([[1,1,1],[1,1,1][1,1,1]])

        if (len(resolution) > 1):
            img = np.array(image, dtype = np.uint8)
            img.resize([resolution[0], resolution[1], 3])
            img = np.rot90(img,2)
            img = np.fliplr(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

            # medianBlur filter - to smooth out an image
            blur_image = cv.medianBlur(img, 15)

            gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _,binary_image = cv.threshold(gray_image, 128, 255, cv.THRESH_BINARY)
            # This line works as an automatic binarizator (finding an umbral by itself)
            # _,binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

            # dilate filter - to dilate an image
            dilate_image = cv.dilate(binary_image, kernelD, iterations = 1)

            # erode filter - to erode an image
            erode_image = cv.erode(binary_image, kernelD, iterations=2)

            # Others import filters to take into account
            img_open = cv.morphologyEx(binary_image,cv.MORPH_OPEN,kernelB, iterations=1)
            img_close = cv.morphologyEx(binary_image,cv.MORPH_CLOSE,kernelB, iterations=1)
            img_gradient = cv.morphologyEx(binary_image, cv.MORPH_GRADIENT, kernelB, iterations=1)

            # cv.imshow("Original Image", img)
            # cv.imshow("Dilate Image",  dilate_image)
            # cv.imshow("Blur Image", blur_image)
            cv.imshow("Binary Image", binary_image)
            cv.imshow("Erode Image", binary_image)
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

def getting_started_contours():
    while (True):
        _, resolution, image = sim.simxGetVisionSensorImage(
            clientID,
            camera,
            0,
            sim.simx_opmode_buffer
        )

        if (len(resolution) > 1):
            img = np.array(image, dtype = np.uint8)
            img.resize([resolution[0], resolution[1], 3])
            img = np.rot90(img,2)
            img = np.fliplr(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

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

            const = 0.5265*54289.0

            for cnt in contours:
                area = cv.contourArea(cnt)

                if (area > 0):
                    M = cv.moments(cnt)
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    cv.drawContours(contour_image, cnt, -1, (255,0,0), 2)
                    cv.circle(contour_image, (cx,cy), 1, (0,255,0), 2)
            print(f"The x-distance is {const/area} m")

            cv.imshow("Binary Image", contour_image)
            cv.imshow("Original Image", img)

        if (cv.waitKey(1) & 0xFF == ord('q')):
            break




getting_started_contours()


# End connexion 
sim.simxFinish(-1)

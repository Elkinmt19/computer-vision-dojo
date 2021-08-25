# Built-in imports 
import sys 

# External imports 
import cv2 as cv
import numpy as np

# Own imports 
import sim
import robotnik_class as rn


def main():
    # End connexion 
    sim.simxFinish(-1)

    # Create new connexion
    clientID = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)

    if (clientID != -1):
        print("Connexion OK")
    else:
        print("Fatal error - No connexion")

    robot = rn.RobotnikClass(clientID)    

    while (True):
        robot.camera_buffer()
        if (len(robot.resolution) > 1):
            img = np.array(robot.image, dtype = np.uint8)
            img.resize([robot.resolution[0], robot.resolution[1], 3])
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

            CONST_INV = 0.5265*54289.0

            for cnt in contours:
                area = cv.contourArea(cnt)

                if (area > 0):
                    M = cv.moments(cnt)
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    cv.drawContours(contour_image, cnt, -1, (255,0,0), 2)
                    cv.circle(contour_image, (cx,cy), 1, (0,255,0), 2)
            print(f"The x-distance is {CONST_INV/area} m")

            cv.imshow("Binary Image", contour_image)
            cv.imshow("Original Image", img)

        if (cv.waitKey(1) & 0xFF == ord('q')):
            break
    # End connexion 
    sim.simxFinish(-1)


if __name__ == "__main__":
    sys.exit(main())
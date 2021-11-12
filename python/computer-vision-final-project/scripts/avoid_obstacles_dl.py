# Built-in imports 
import os
import sys 
import threading as thr
from time import time

# External imports 
import cv2 as cv
import numpy as np

# Own imports 
import sim
import kuka_youbot_autopilot_class as kb

# My Own imports
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

class AvoidObstaclesDL:
    def __init__(self, robot, show_cameras):
        # Define the robot's parameters
        self.robot = robot
        self.CONS_DIST = 20985.274362037777
        self.ANGLE = 0.4333843118451189

        # Define variables for the deeplearning object detection algorithm
        self.yolo_files_path = os.path.join(ASSETS_FOLDER, "yolo")
        self.load_yolo_algorithm()
        self.show_cameras = show_cameras

        # Define variables for the avoid obstacle algorithm 
        self.cameras = {
            "x_locations": [],
            "distances": [],
            "objects": []
        }

        # Define thread's variables for the object detection implementation 
        self.cameras_threads = [thr.Thread(target=self.detect_objects(x)) for x in iter(range(4))]

    def load_yolo_algorithm(self):
        # First it is mandatory to load the yolo algorithm
        self.net = cv.dnn.readNet(
            f"{self.yolo_files_path}/yolov3_custom_last_v4.weights",
            f"{self.yolo_files_path}/yolov3_custom.cfg"
        )

        self.classes = []
        with open(f"{self.yolo_files_path}/obj.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect_objects(self, cam):
        try:
            # Preprocessing of the image
            img = np.array(self.robot.image[cam], dtype = np.uint8)
            img.resize([self.robot.resolution[cam][0], self.robot.resolution[cam][1], 3])
            img = np.rot90(img,2)
            img = np.fliplr(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

            o_h, o_w = img.shape[:2]
            img = cv.resize(img, None, fx=0.4, fy=0.4)
            height, width = img.shape[:2]

            # Perform the object detection
            #! yolo just accepts three sizes [(320,320),(609,609),(416,416)] 
            blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers) # This variable has all the information of the objects detected

            # Show the result on the screen 
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if (confidence) > 0.5: # Just show the detections with an accuracy greater than 50%
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Sometimes happen that we got more than 1 single box per object, to avoid this the following function is used
            indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Define some important variables for the avoid obstacle algorithm
            distances = []
            x_locations = []
            objects = []

            # Get the objects features and show the image with the objects detected
            font = cv.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    color = self.colors[i]
                    cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    distances.append(self.CONS_DIST/((x + w)*(y + h)))
                    x_locations.append(x + w/2 - width/2)
                    objects.append(label)
                    cv.putText(img, label, (x, y + 30), font, 1, color, 2)
            # img = cv.resize(img, [o_h,o_w]) 

            if (self.show_cameras):
                cv.imshow(f"Image {cam}", img)
            self.cameras["distances"].append(distances)
            self.cameras["x_locations"].append(x_locations)
            self.cameras["objects"].append(objects)
        except:
            print("An error just happen!!!")

def main():
    # End connection 
    sim.simxFinish(-1)

    # Create new connection
    clientID = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)

    if (clientID != -1):
        print("Connection OK")
    else:
        print("Fatal error - No connection")

    robot = kb.KukaYouBotClass(clientID)


    while (1):
        t0 = time()
        robot.camera_buffer()
        controller = AvoidObstaclesDL(robot, True)
        for cm in controller.cameras_threads:
            cm.start()

        for cm in controller.cameras_threads:
            cm.join()
        
        print(controller.cameras)

        if (cv.waitKey(1) & 0xFF == ord('q')):
            print("We are MELOS!!!")
            break
        print(time() - t0)
    # End connection 
    sim.simxFinish(-1)

if __name__ == "__main__":
    sys.exit(main())
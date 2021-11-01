# Built-in imports 
import os

# External imports 
import cv2 as cv
import numpy as np

# My Own imports
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

yolo_files_path = os.path.join(
        ASSETS_FOLDER, "yolo")

# First it is mandatory to load the yolo algorithm
net = cv.dnn.readNet(f"{yolo_files_path}/yolov3_custom_last_v2.weights", f"{yolo_files_path}/yolov3_custom.cfg")
classes = []
with open(f"{yolo_files_path}/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the image where the object detection is going to be perform
img = cv.imread("test_image.jpg",1)
img = cv.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Perform the object detection
#! yolo just accepts three sizes [(320,320),(609,609),(416,416)] 
blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers) # This variable has all the information of the objects detected

# Show the result on the screen 
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5: # Just show the detections with an accuracy greater than 50%
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

# Get the objects features and show the image with the objects detected
font = cv.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.putText(img, label, (x, y + 30), font, 3, color, 3)
cv.imshow("Image", img)
cv.waitKey(0)
cv.destroyAllWindows()
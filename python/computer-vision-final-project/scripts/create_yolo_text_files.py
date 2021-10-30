# Built-in imports
import os
import sys
import csv
import shutil

# External imports 
import cv2

class CreateYoloTextFiles:
    def __init__(self, show_objects):
        """
        This is python class that allows to build a Yolo-format dataset base on 
        a TensorFlow-format dataset, It has some simple methods that read some 
        csv files that contains critical information about the object and their 
        features.
        :param: show_objects: This is a simple flag that allows to show the images
        that the program is working with while the algorithm is running.
        """
        self.classes = [
            "person",
            "mannequin",
            "dog",
            "plant",
            "table",
            "chair",
            "tree",
            "washbasin",
            "bathroom",
            "stairs",
            "laptop",
            "container"
        ]

        self.show_objects = show_objects

        self.csv_file = "cameras_data_training.csv"
        self.img_directory = "images-to-training\\"

        self.yolo_directory = "images-to-training-yolo-format\\"
        self.yolo_file_list = 'yolo_training_list.txt'
        self.path = os.getcwd() + '\\'

    def convert(self, size, box, label):
        """
        This is a simple python method that allows to get the coordinates 
        for the respective rectangules that are going to capture the
        objects of the classes of the trained model.
        """
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (label, x, y, w, h)

    def read_csv_file(self):
        """
        This is python method that goes through the whole TensorFlow-format dataset 
        in order to build a Yolo-format dataset, this method analyze every sample of 
        the dataset, getting the most important and relevant features of the different
        objects that were detected when the labeling was made.
        """    
        with open(self.csv_file) as File:
            reader = csv.reader(File)
            for row in reader:
                File = f"{self.img_directory}{row[0]}"
                if row[0] != 'filename':
                    if os.path.isfile(File):
                        img = cv2.imread(File, 1)
                        H, W = img.shape[:2]

                        # Map the string classes into numeric values 
                        labeled = 0
                        for i in iter(range(len(self.classes))):
                            if (row[3] == self.classes[i]):
                                labeled =  i
                            
                        box = (float(row[4]), float(row[6]), float(row[5]), float(row[7]))
                        bb = self.convert((W, H), box, labeled)
                        
                        # Write the features of the objects of the image into a text file
                        self.write_csv_file(f"{self.yolo_directory}{row[0]}", bb)

                        if not os.path.isfile(f"{self.yolo_directory}{row[0]}"):
                            shutil.copy(File, f"{self.yolo_directory}{row[0]}")
                            self.write_csv_file(self.yolo_file_list, [f"{self.path}{self.yolo_directory}{row[0]}"])

                        if (self.show_objects):    
                            cv2.rectangle(img, (int(row[4]), int(row[5])), (int(row[6]), int(row[7])), (255, 0, 0), 2)
                            cv2.circle(img, (int(bb[1]*W), int(bb[2]*H)), 5, (0, 255, 0), -1)
                            cv2.imshow("image", cv2.resize(img, (640, 480)))                  
                        
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            cv2.destroyAllWindows()
            
    def write_csv_file(self, text_file, data):
        """
        This is a python method that allows to write the information of a specific
        sample of the dataset into a text file in order to build the Yolo-format 
        dataset at the end of the program's execution.
        """
        text_file = text_file[:text_file.find('.')] + '.txt'
        my_file = open(text_file, 'a', newline = '')

        with my_file:
            writer = csv.writer(my_file, delimiter=' ')
            writer.writerow(data)
        print("Writing complete!!!")

    def build_yolo_dataset(self):
        if not os.path.exists(self.yolo_directory):
            os.mkdir(self.yolo_directory)
        self.read_csv_file()
        print("Build process done!!!")

def main():
    _ = CreateYoloTextFiles(False).build_yolo_dataset()

if __name__ == "__main__":
    sys.exit(main())

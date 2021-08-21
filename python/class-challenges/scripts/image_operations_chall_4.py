# Built-int imports 
import os
import sys

# External imports
import cv2 as cv
import numpy as np

# My own imports 
import image_segmentation as imgs
import image_analysis as imga
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

class ImageOperations:
    def __init__(self):
        self.get_images()

    def get_image_path(self, image_name):
        image_relative_path = os.path.join(
        ASSETS_FOLDER, "imgs", image_name)
        return image_relative_path

    def get_images(self):
        # Upload the images for the first point
        self.pills = [
            cv.imread(self.get_image_path(f"pills_{x}.png")) for x in iter(range(1,3))
        ]

        # Upload the images for the second point
        self.frames = [
            cv.imread(self.get_image_path(f"frame_{x}.png")) for x in iter(range(1,31,29))
        ]

    def get_values_pills_segmentation(self):
        """
        This is a simple method to find the umbral's values 
        for the segmentation of the pill images for the first
        point of the challenge.
        """
        for img in self.pills:
            color_seg = imgs.ImageSegmentation(img)
            color_seg.color_segmentation("HSV")

    def count_frequencies(self):
        labels = list()
        frequency = list()

        # Mark all array elements as not visited
        rows, cols = self.labels_img.shape[:2]
        visited = np.zeros((
                rows,
                cols
        ), dtype=bool)
    
        # Traverse through array elements
        # and count frequencies
        for i in iter(range(rows)):
            for j in iter(range(cols)):
                # Skip this element if already
                # processed
                if (visited[i,j] == True or self.labels_img[i,j] == 0):
                    continue
        
                # Count frequency
                count = 1
                for w in iter(range(rows)):
                    if (w == i):
                        start_condition = j + 1
                    else:
                        start_condition = 0
                    for k in iter(range(start_condition, cols)):
                        if (self.labels_img[i,j] == self.labels_img[w,k]):
                            visited[w,k] = True
                            count += 1
                
                labels.append(self.labels_img[i,j])
                frequency.append(count)
        self.labels_freq = dict(zip(labels, frequency))
    
    def count_pills(self):
        segmented_pills = list()
        upper_values = np.array([159, 164, 255])
        lower_values = np.array([114, 73, 86])

        for img in self.pills:
            segmented_pills.append(cv.inRange(
                cv.cvtColor(img, cv.COLOR_BGR2HSV),
                lower_values,
                upper_values
            ))
        
        subtracted_pill = cv.subtract(
            segmented_pills[0],
            segmented_pills[1]
        )

        num_labels, self.labels_img = cv.connectedComponents(subtracted_pill)
        print(f"Number of labels: {num_labels}")
        self.count_frequencies()
        cv.imshow("Subtracted pill", subtracted_pill)
        cv.waitKey(0)



def main():
    img_operator = ImageOperations()
    img_operator.count_pills()

if __name__=='__main__':
    sys.exit(main())
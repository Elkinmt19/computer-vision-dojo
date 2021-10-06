import os
import cv2 as cv
import numpy as np

# My Own imports
import get_path_assests_folder as gpaf

# Get assets folder in repo for the samples
ASSETS_FOLDER = gpaf.get_assets_folder_path()

def extract_features(image, vector_size=32):
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv.error as e:
        print ('Error: ', e)
        return None

    return dsc

path_characters = os.path.join(
    ASSETS_FOLDER, "imgs", f"letters/A/4 (2).jpg")

image = cv.imread(path_characters, 1)

dia = np.diagonal(image)

print(dia.shape)

features = extract_features(image)
print(features[0])

cv.imshow("Image test", image)
cv.waitKey(0)
import os
import cv2 as cv
import numpy as np
from glob import glob

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


def corner_detection(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    # find Harris corners
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    dst = cv.dilate(dst,None)
    _, dst = cv.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)

    men1_res = float(sum(list(res[:,1]))/len(list(res[:,1])))
    men2_res = float(sum(list(res[:,2]))/len(list(res[:,2])))
    men3_res = float(sum(list(res[:,3]))/len(list(res[:,3])))
    men4_res = float(sum(list(res[:,4]))/len(list(res[:,4])))


    #image[res[:,1],res[:,0]]=[0,0,255]
    #image[res[:,3],res[:,2]] = [0,255,0]
    # cv.imshow('subpixel5.png',image)
    # cv.waitKey(0)

    return men1_res, men2_res, men3_res, men4_res

def corner_detection_2(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    image[dst>0.01*dst.max()]=[0,0,255]
    key_variable = dst>0.01*dst.max()
    
    x = list(np.where(key_variable == True)[0])
    y = list(np.where(key_variable == True)[1])

    xmean = sum(x)/len(x)
    ymean = sum(y)/len(y)

    # cv.imshow('dst',image)
    # cv.waitKey(0)
    return xmean, ymean

character_list = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "P",
    "R",
    "S",
    "T",
    "V",
    "X",
    "Z"
]


for char in character_list:
    path_characters = os.path.join(
        ASSETS_FOLDER, "imgs", f"letters/{char}/*.jpg")

    list_paths = glob(path_characters)
    images = [cv.imread(x,1) for x in list_paths]

    image = cv.imread(path_characters, 1)

    list_count = list()
    list_count1 = list()

    # list1 = list()
    # list2 = list()
    # list3 = list()
    # list4 = list()

    for img in images:
        x, y = corner_detection_2(img.copy())
        list_count.append(x)
        list_count1.append(y)

        # x, y, z, w = corner_detection(img.copy())
        # print(x,y,z,w)
        # list1.append(x)
        # list2.append(y)
        # list3.append(z)
        # list4.append(w)

    # try:
    #     print(f"Mean value1 {char}: {sum(list1)/len(list1)}")
    #     print(f"Mean value2 {char}: {sum(list2)/len(list2)}")
    #     print(f"Mean valu3 {char}: {sum(list3)/len(list3)}")
    #     print(f"Mean value4 {char}: {sum(list4)/len(list4)}")
    # except:
    #     print("Something wrong just happend")

    print(f"Mean value {char}: {sum(list_count)/len(list_count)}")
    print(f"Mean value1 {char}: {sum(list_count1)/len(list_count1)}")

    # features = extract_features(image)
    # print(features[0])

    # cv.imshow("Image test", image)
    # cv.waitKey(0)
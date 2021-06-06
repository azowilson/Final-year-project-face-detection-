import numpy as np
from CV2HOG import hog_feature
import os
from skimage.io import imread
from skimage import color
import cv2
from tqdm import tqdm
def get_hog(trainPath="", categories=["negative_samples","Face"], no_negative = 70000, no_positive=6700):
    print("Start getting hog features...")

    feature_collection=[]
    #files = os.listdir(trainPath)

    imgPathDict = {}
    filePath = []
    imgValue = []
    labels =[]
    Features = []
    #new dataset dimension 36x36
    # img = cv2.imread(r"E:\dataset\Face\caltech_web_crop_00001.jpg", 0)
    # d = img.shape
    # print(d)
    count = 0
    for category in categories:

        # positive sample path and negative sample path
        path = os.path.join(trainPath, category)
        files = os.listdir(path)
        if category == "negative_samples":
            files = files[0:no_negative]

        #label the class with index (0, 1)
        #non-face: 0
        #face: 1

        label = categories.index(category)
        #initialize the img[catagory] list
        #modify the sample number
        # no of positive samples：6713
        if category == "Face":
            #sample_no = 4000
            files = files[0:no_positive]

        imgPathDict[category] = None


        for file in files:
            filePath.append(file)

            #pass imgValue to the imgDict
            imgPathDict[category] = filePath
            #empty the imgValue[]
            filePath=[]
            for key, value in enumerate(imgPathDict[category]):
                count += 1
                #if sample_no != 0:
                imgValue = cv2.imread(os.path.join(path, value),0)
                imgValue = cv2.resize(imgValue,(30,30))

                    #increases 2 times the negative sample size
                #if category == "negative_samples":
                imgMirror = imgValue[:,::-1]
                hog_mirr_features = hog_feature().compute(imgMirror).flatten()
                Features.append([hog_mirr_features, label])

                hog_features = hog_feature().compute(imgValue).flatten()
                    #hog_features = composed_hog(imgValue).flatten()

                #print(hog_features.shape)
                    #hog_mirr_features = composed_hog(imgMirror).flatten()
                Features.append([hog_features,label])


                    #sample_no -= 1



   # Features = np.asarray(Features)
    print("All done!!!")
    print("Total number of samples: ",count)
    #print(Features.shape)

    return Features


#get_hog("E:\dataset",no_negative=20000)




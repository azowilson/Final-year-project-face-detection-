import time
import numpy as np
import cv2
from CV2HOG import hog_feature, composed_hog
from non_max_suppression import non_max_supperssion
import os
from SVM import SVM
import pickle


trainPath = r"C:\Users\leung\Desktop\SVM\code\GUI"
def img_resize(img, width = None, height=None, internal= cv2.INTER_AREA):
    dim = None
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    (h, w) = img.shape[0],img.shape[1]
    if width==None and height==None:
        return img

    if width != None:
        ratio = width / float(w)
        dim=(width, int(h*ratio))
    else:
        ratio = height/float(h)
        dim=(int(w*ratio), height)
    resizeImg = cv2.resize(img, dim, interpolation=internal)
    return resizeImg

def check_model(model_path=""):
    if os.path.exists(model_path):
        model = pickle.load(open(model_path,"rb"))
        return model
    else:
        return None

def face_detector():
    #model_path = r"model\face_detector.model"
    #smaller
    #test 6 #lambda = 0.01
    #test 8 #lambda = 0.1
    #test 9 #lambda = 0.001
    #test 10 #lambda = 0.001 iteration = 200
    model_path = ".\model\svm_model_test10.model"
    # lda = LDA(2)
    # lda.fit(feat, labels)
    # X_projected = lda.transform(feat)
    if check_model(model_path)==None:

        features = get_hog(trainPath, no_negative=70000, no_positive=6700)
        feat = []
        labels = []
        for feature, y in features:
            feat.append(feature)
            labels.append(y)
        feat = np.array(feat)
        labels = np.array(labels)
        model = SVM()
        model.fit(feat, labels)
        pickle.dump(model, open(model_path, "wb"))

    else:
        model = check_model(model_path)
    # features = lda.transform(features)
    #model, conf_features = SVM_classifier(features)
    # print(features[1])
    # model = SVM()
    # model.fit(feat, labels)
    model.showConverge()
    testPath = r"./testimg"
    testImg = os.listdir(testPath)
    print(testImg)
    # print(len(testImg))
    # print(os.path.join(testPath, testImg[1]))
    # initializing variables
    bboxes = np.zeros([0, 4])
    confidences = np.zeros([0, 1])
    image_ids = np.zeros([0, 1])

    #confidences = model.decision_function(conf_features)
    # print(confidences)
    # print(len(confidences))
    # step size 8 = 63.6; step size 6 = 77.3; step size 4 = 88.6; step size 3 = 88.6 step 5 =  79.5%
    # h=10, v=3 acc=68.1 fp = 3
    # h=8, v=3, acc=79.5 fp = 1

    # h_stepSize = 30
    # v_stepSize = 30
    # downSample = 0.7

    downSample = 0.95
    h_stepSize = 2
    v_stepSize = 2

    #1: 1.6, 2->1.6
    threshold = 0.8
    file = open("boxes.txt", "w").close()
    for i in range(len(testImg)):

        t0 = time.time()
        scale = 1
        curr_bbox = np.zeros([0, 5])
        curr_confidence = np.zeros([0, 1])
        curr_img_idx = np.zeros([0, 1])
        img = cv2.imread(os.path.join(testPath, testImg[i]), 0)
        resizeImg = img_resize(img, width=600)
        colorImg = cv2.imread(os.path.join(testPath, testImg[i]))
        colorImg = img_resize(colorImg, width=600)
        H, W = resizeImg.shape  # (height, width)
        print(H, W)
        # find the shortest one
        minLen = min(H, W)

        # loop one image
        winSize = 30
        cellSize = 2
        cellNum = 5
        # print(winSize)H
        # if the shorter side larger than the frame size (30, 30)
        while minLen * scale >= winSize * 3:
            imgResize = cv2.resize(resizeImg, (int(W * scale), int(H * scale)))
            # hog_f = composed_hog(imgResize)
            for i in range(0, int(H * scale), v_stepSize):
                for j in range(0, int(W * scale), h_stepSize):
                    movingWindow = imgResize[i:i + winSize, j:j + winSize]
                    if movingWindow.shape[0] != winSize or movingWindow.shape[1] != winSize:
                        continue

                    #savemodel no need reshape
                    hogFrame = hog_feature().compute(movingWindow).flatten()

                    # hogFrame = movingWindow.flatten().reshape(1,-1)
                    # prediction = model.predict(hogFrame)
                    # print(prediction)
                    # confidence = model.predict(hogFrame)
                    #confidence = model.decision_function(hogFrame)
                    # print(confidence)
                    confidence = model.predict(hogFrame)
                    #confidence = model.decision_function(hogFrame)
                    if confidence > threshold:
                        # print(confidence)

                        topLeftX = int(j / scale)
                        topLeftY = int(i / scale)
                        botRightX = int((j + winSize) / scale)
                        botRightY = int((i + winSize) / scale)
                        print("(%s %s %s %s)" % (topLeftX, topLeftY, botRightX, botRightY))

                        confy = np.round(confidence, 2).astype(float)
                        confy = confy.item()

                        # file = open("boxes.txt", "a")
                        # file.write("%s,%s,%s,%s" % (topLeftX,topLeftY,botRightX,botRightY))
                        # file.write("\n")
                        box = np.array([[topLeftX, topLeftY, botRightX, botRightY, confy]])
                        curr_bbox = np.concatenate((curr_bbox, box), 0)
                        #curr_confidence = np.concatenate((curr_confidence, [confidence]), 0)
                        # test image id
                        # curr_img_idx = np.concatenate((curr_img_idx, [[testPath[i]]]), 0)

                    # clone = imgResize.copy()
                    # cv2.rectangle(clone, (int(j / scale), int(i / scale)), (int((j + winSize) / scale),
                    #                                                         int((i + winSize) / scale)),
                    #               (255, 0, 0), 2)
                    # cv2.imshow("Window", clone)
                    # cv2.waitKey(1)
                    # time.sleep(0.00025)
            scale = scale * downSample
            # print(scale)

        t1 = time.time()
        print("time used: ", (t1 - t0))

        if curr_bbox.any():
            bboxes = non_max_supperssion(curr_bbox)
            print("Total detections: ", bboxes.shape)
            # x1 = bboxes[:, 0]
            # y1 = bboxes[:, 1]
            # x2 = bboxes[:, 2]
            # y2 = bboxes[:, 3]
            x1 = bboxes[:, 0].astype(int)
            y1 = bboxes[:, 1].astype(int)
            x2 = bboxes[:, 2].astype(int)
            y2 = bboxes[:, 3].astype(int)
            conf =  bboxes[:, 4]
        # x1 = curr_bbox[:,0]
        # y1 = curr_bbox[:,1]
        # x2 = curr_bbox[:,2]
        # y2 = curr_bbox[:,3]

        for j in range(len(x2)):
            cv2.rectangle(colorImg, (x1[j], y1[j]), (x2[j], y2[j]), (0, 0, 255), 2)
            cv2.putText(colorImg, str(conf[j]), (x1[j], y2[j]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
        cv2.imshow("Window", colorImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


face_detector()

import numpy as np
import pandas as pd
from tqdm import tqdm
##test required libs###
import cmath
import os
import cv2
from CV2HOG import hog_feature
import matplotlib.pyplot as plt
from LDA import LDA
import random
#######################
class SVM:
    def __init__(self, learning_rate = 0.0001, lambda_param = 0.001, n_iters=200):
        self.lr = learning_rate
        self.lambda_para = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.weight = []

    def fit(self, X, y):
        print("Start training SVM...")
        y_ = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape
        print(n_features)
        self.w = np.zeros(n_features)
        self.b = 0
        #gradient
        for _ in tqdm(range(self.n_iters)):
            for idx, x_i in enumerate(X):

                condition = y_[idx] * (np.dot(x_i, self.w)- self.b) > 1
                if condition:
                    self.w = self.w - self.lr * (self.lambda_para * self.w) #update to weight
                else:
                    self.w = self.w - self.lr *(self.lambda_para * self.w - np.dot(x_i, y_[idx]))
                    self.b = self.b - self.lr * y_[idx]

                #print(np.dot(x_i, self.w) - self.b)
            self.weight.append(self.w)
        print("Train completed...")



    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        #return np.sign(linear_output)
        return linear_output

    def decision_function(self, x_test):
        pass

    def showConverge(self):
        rangeNum = list(range(0,10000))
        randomNum = random.sample(rangeNum,10)
        for i in randomNum:
            x = list(range(0, 200))
            ww = np.array(self.weight)
            y = ww[:,i][0:200]
            plt.plot(x, y)

        plt.xlabel('no. of iterations')
        plt.ylabel('weight')
        plt.title('training convergence')
        plt.show()


# root = r"E:\dataset"
# categories = ["negative", "positive"]
# imgPathDir={}
# filePath = []
# data = []
# for category in categories:
#     path = os.path.join(root, category)
#     files = os.listdir(path)
#     label = categories.index(category)
#
#     imgPathDir[category]=None
#     for file in files:
#         filePath.append(file)
#         imgPathDir[category] = filePath
#         filePath = []
#         for key, value in enumerate(imgPathDir[category]):
#             imgValue = cv2.imread(os.path.join(path,value), 0)
#             imgValue = cv2.resize(imgValue,(30,30))
#
#             #imgValue = imgValue.flatten()
#             #data.append([imgValue, label])
#             hog = hog_feature().compute(imgValue).flatten()
#             data.append([hog, label])
#
# features =[]
# labels =[]
# for feature, y in data:
#     features.append(feature)
#     labels.append(y)
# features = np.array(features)
# labels = np.array(labels)
# #print(labels)
# lda = LDA(2)
# lda.fit(features,labels)
# X_projected = lda.transform(features)
#
# print('Shape of X:', features.shape)
# print('Shape of transformed X:', X_projected.shape)
#
# # x1 = X_projected[:, 0]
# # # x2 = X_projected[:, 1]
# # #
# # # plt.scatter(x1, x2,
# # #         c=labels, edgecolor='none', alpha=0.8,
# # #         cmap=plt.cm.get_cmap('viridis', 3))
# # #
# # # plt.xlabel('Linear Discriminant 1')
# # # plt.ylabel('Linear Discriminant 2')
# # # plt.colorbar()
# # # plt.show()
# y = np.where(labels==0, -1, 1)
# clf = SVM()
# clf.fit(X_projected, y)
# print(clf.w, clf.b)
# complexResult = clf.predict(X_projected[77])
# print(complexResult)
# print(type(complexResult))
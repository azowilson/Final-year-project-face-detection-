import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
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
        print("no. of features per img: "+ str(n_features))
        print("no. of training iterations: "+ str(self.n_iters))
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



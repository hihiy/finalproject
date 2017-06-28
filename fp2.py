import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle
from liblinearutil import *
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.model_selection import train_test_split


class M3Model():

    def __init__(self,x_train,split = 2):
        self.x_train = x_train
        self.split = split
        self.m12 = []
        self.m13 = []
        self.m23 = []

    def min_model(self):
        self.m12 = self.min_max_model(self.x_train[0], self.x_train[1])

    def min_max_model(self,c1,c2):
        c1_y = np.ones(c1.shape[0]).reshape(c1.shape[0], 1)
        c2_y = np.zeros(c2.shape[0]).reshape(c2.shape[0], 1)
        # y = np.concatenate([c1_y, c2_y])
        # x = np.concatenate([c1,c2])
        c1_x_split = []
        c2_x_split = []
        c1_y_split = []
        c2_y_split = []
        tx1 = c1.copy()
        tx2 = c2.copy()
        ty1 = c1_y.copy()
        ty2 = c2_y.copy()
        for i in range(self.split-1):
            x1, tx1, y1, ty1 = train_test_split(tx1, ty1, test_size=1.-1./(self.split-i))
            c1_x_split.append(x1)
            c1_y_split.append(y1)
            x2, tx2, y2, ty2 = train_test_split(tx2, ty2, test_size=1. - 1. / (self.split - i))
            c2_x_split.append(x2)
            c2_y_split.append(y2)

        minmaxmatrix_x = [[] for i in range(self.split)]
        minmaxmatrix_y = [[] for i in range(self.split)]
        for i in range(self.split):
            temprx = []
            tempry = []
            for j in range(self.split):
                tempx = np.concatenate([c1_x_split[i], c2_x_split[j]])
                temprx.append(tempx)
                tempy = np.concatenate([c1_y_split[i], c2_y_split[j]])
                tempry.append(tempy)
            minmaxmatrix_x[i] = temprx
            minmaxmatrix_y[i] = tempry

        minmaxmodel = []
        

        # temp = np.hstack([x, y]).copy()
        # np.random.shuffle(temp)
        # xx = np.delete(temp, -1, 1)
        # yy = np.delete(temp, np.s_[0:310], 1).reshape(y.shape[0])
        x1, x2, y1, y2 = train_test_split(x, y, test_size=0.5)

    def train(self):
        self.min_model()

    def predict(self):
        pass


def main():
    with open('eeg_class.pkl', 'rb') as f:
        x_train_class = pickle.load(f)
        x_test_class = pickle.load(f)





if __name__ == '__main__':
    main()
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
        y = np.concatenate([c1_y, c2_y])
        x = np.concatenate([c1,c2])
        c1_x_split = []
        c2_x_split = []
        c1_y_split = []
        c2_y_split = []
        for i in range(self.split):
            pass

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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle
from liblinearutil import *
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.model_selection import train_test_split




def main():
    with open('eeg_class.pkl', 'rb') as f:
        x_train_class = pickle.load(f)
        x_test_class = pickle.load(f)





if __name__ == '__main__':
    main()
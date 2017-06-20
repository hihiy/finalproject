# import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pickle
from liblinearutil import *
# ori_data = sio.loadmat("EEG.mat")
# x_train = ori_data['X_Train'][0]
# y_train = ori_data['Y_Train'][0]
# x_test = ori_data['X_Test'][0]
# y_test = ori_data['Y_Test'][0]
#
# with open('eeg.pkl', 'wb') as f:
#     pickle.dump(x_train, f, True)
#     pickle.dump(y_train, f, True)
#     pickle.dump(x_test, f, True)
#     pickle.dump(y_test, f, True)

with open('eeg.pkl', 'rb') as f:
    x_train = pickle.load( f)
    y_train = pickle.load(f)
    x_test = pickle.load( f)
    y_test = pickle.load( f)


total=0
ave=0
for i in range(45):
    y = y_train[i].reshape(2010)
    x = x_train[i]
    prob  = problem(y, x)
    param = parameter('-s 0 -c 2 -B 1 -q')
    m = train(prob, param)

    yy = y_test[i].reshape(1384)
    xx = x_test[i]
    p_label, p_acc, p_val = predict(yy, xx, m, '-b 0')
    ACC, MSE, SCC = evaluations(yy, p_label)

    total += ACC

print(total/45)
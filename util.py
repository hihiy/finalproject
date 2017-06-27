import scipy.io as sio
import pickle
import numpy as np

ori_data = sio.loadmat("EEG.mat")
x_train = ori_data['X_Train'][0]
y_train = ori_data['Y_Train'][0]
x_test = ori_data['X_Test'][0]
y_test = ori_data['Y_Test'][0]

# with open('eeg.pkl', 'wb') as f:
#     pickle.dump(x_train, f, True)
#     pickle.dump(y_train, f, True)
#     pickle.dump(x_test, f, True)
#     pickle.dump(y_test, f, True)


x_train_class =[]
x_test_class =[]
for i in range(45):
    temp = [[] for k in range(3)]

    for j in range(len(y_train[i])):
        if y_train[i][j][0] == -1:
            temp[0].append(x_train[i][j])
        elif y_train[i][j][0] == 0:
            temp[1].append(x_train[i][j])
        elif y_train[i][j][0] == 1:
            temp[2].append(x_train[i][j])
    for j in range(3):
        temp[j] = np.array(temp[j])
    temp = np.array(temp)
    x_train_class.append(temp)

    temp = [[] for k in range(3)]

    for j in range(len(y_test[i])):
        if y_test[i][j][0] == -1:
            temp[0].append(x_test[i][j])
        elif y_test[i][j][0] == 0:
            temp[1].append(x_test[i][j])
        elif y_test[i][j][0] == 1:
            temp[2].append(x_test[i][j])
    for j in range(3):
        temp[j] = np.array(temp[j])
    temp = np.array(temp)
    x_test_class.append(temp)

with open('eeg_class.pkl', 'wb') as f:
    pickle.dump(x_train_class, f, True)
    pickle.dump(x_test_class, f, True)


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle
from liblinearutil import *
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.model_selection import train_test_split



with open('eeg.pkl', 'rb') as f:
    x_train = pickle.load( f)
    y_train = pickle.load(f)
    x_test = pickle.load( f)
    y_test = pickle.load( f)


x1,x2,y1,y2 = train_test_split(x_test,y_test,test_size=0.5)


total=0
ave=0
cnf_matrix = np.zeros((3,3))
for i in range(45):
    y = y_train[i].reshape(y_train[i].shape[0])
    x = x_train[i]
    prob  = problem(y, x)
    param = parameter('-s 0 -c 4 -B 1 -q')
    m = train(prob, param)
    # save_model('./model2/-s0-c4-B1-q-TRAIL'+str(i)+'.model', m)
    yy = y_test[i].reshape(y_test[i].shape[0])
    xx = x_test[i]
    p_label, p_acc, p_val = predict(yy, xx, m, '-b 0')
    temp_cnf_matrix = confusion_matrix(yy, p_label)
    np.set_printoptions(precision=2)
    ACC, MSE, SCC = evaluations(yy, p_label)

    total += ACC
    cnf_matrix += temp_cnf_matrix

print(total/45)
print(cnf_matrix)

class_names = ['-1','0','1']
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
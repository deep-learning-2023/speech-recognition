import numpy as np
import pandas as pd
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
import random
import os
from keras.utils.np_utils import to_categorical   
from itertools import repeat
import matplotlib.pyplot as plt
import seaborn as sn

def print_data_description(x, y):
    print("x_train shape: ", x.shape)
    print("y_train shape: ", y.shape)
    y_dict = {}
    for y_value in y:
        if not y_value in list(y_dict.keys()):
            y_dict[int(y_value)] = 0
        y_dict[int(y_value)] += 1
    print("examples per class: ", y_dict)

def get_accuracy_description(new_model, x_test, y_test, labels):
    result = ""
    y_test_pred = new_model.predict(x_test)
    y_test_pred = np.argmax(y_test_pred, axis=1) 
    classes = np.max(y_test) + 1
    TN = list(repeat(0, classes))
    TP = list(repeat(0, classes))
    FN = list(repeat(0, classes))
    FP = list(repeat(0, classes))
    CC = list(repeat(0, classes)) # class count, how many examples per class
    for i in range(0, len(y_test_pred)):
        c = int(y_test[i])
        CC[c] += 1
        if c == y_test_pred[i]:
            TP[c] += 1
            for j in range(0, 8):
                if j != c: TN[j] += 1
            else:
                FP[c] += 1
                FN[int(y_test_pred[i])] += 1
    for i in range(1, classes):
        if TN[i] + TP[i] + FN[i] + FP[i] == 0: 
            result += str(i) +"___"
            continue
        acc = round((TN[i] + TP[i])/(TN[i] + TP[i] + FN[i] + FP[i]),2)
        acc_local = round(TP[i]/CC[i],2)
        result += str(i) +", "+ str(labels[i - 1]) +", "+ str(TN[i])+", "+ str(TP[i])+", "+ str(FN[i])+", "+ str(FP[i])+", "+ " acc: "+", "+ str(acc)+", "+ " acc_local: "+", "+ str(acc_local) + "\n"
    acc_glob = (np.sum(TN) + np.sum(TP))/(np.sum(TN) + np.sum(TP) + np.sum(FN) + np.sum(FP))
    acc_local = []
    for i in range(1, len(TP)):
        if CC[i] > 2:
            acc_local.append(TP[i] / CC[i])
    acc_local = np.mean(acc_local)
    result += "acc global: "+ str(acc_glob)
    result += "acc local: "+ str(acc_local)
    return result

def draw_hist(hist, path):
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    xc = range(0, len(train_acc))
    plt.figure()
    plt.plot(xc, train_acc, label = "train accuracy")
    plt.plot(xc, val_acc, label = "test accuracy")
    plt.legend()
    plt.savefig(path + "\learning_history.png")
    plt.close()

def draw_conf_matrix(new_model, x, y, labels, label, path):
    y_pred = new_model.predict(x)

    y_pred = np.argmax(y_pred, axis=1) 
    cm = confusion_matrix(y, y_pred) 
    cm = np.round(cm / y_pred.shape[0],2)
    df_cm = pd.DataFrame(cm, labels, labels)
    fig = plt.figure(figsize=(16, 14))
    sn.set(font_scale=0.8) 
    s = sn.heatmap(df_cm, annot=True)#, annot_kws={"size": 16}
    s.set(xlabel='Predicted', ylabel='True label')
    #plt.figure(figsize=(32, 32))
    plt.tight_layout()
    plt.savefig(path + "/" +label + "_conf_matrix.png")
    plt.close()
    return "Confusion matrix " + label + " \n" + str(cm)

def save_txt(decription, path):
    with open(path+'/decription.txt', 'w') as f:
        f.write(decription)
    f.close()
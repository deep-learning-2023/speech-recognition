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
import visualization
import set_seed
import models.GRU as GRU
import models.LSTM as LSTM
import models.simplernn as SRNN
import preprocessing


def get_model_summary(model):
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

model_name_save = "architecture_experiment_full_2_512"
path = "./results/"+model_name_save
description = " 1000 examples per class "
if not os.path.exists("./results/"):
     os.mkdir("./results/", 0o666)
if os.path.exists(path):
        print("Experiment with this name already exists")
else:
    os.mkdir(path, 0o666)
raw_x, raw_y, label_coding = preprocessing.get_raw_data('C:\\Uczymy sie\\sem_11\\data\\train\\audio\\', 1000000)
x, y = preprocessing.get_mfcc(raw_x, raw_y)
timestamp = x.shape[-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
models_maangers = [GRU, LSTM, SRNN]
labels = label_coding
architecture_type = [None, [32], [64], [128]]

for repetition in range (0, 5):
    set_seed.set_seed(repetition)
    repetition_path = path + "/repetition_" + str(repetition)
    os.mkdir(repetition_path, 0o666)
    for i in range(0, len(architecture_type)):
        for rnn_layer_mode in [False, True]:
            architecture_add_on = architecture_type[i]
            if rnn_layer_mode and architecture_add_on == None: continue
            if rnn_layer_mode:
                experiment_path = repetition_path + "/rnn_"+str(architecture_add_on)
            else:
                experiment_path = repetition_path + "/dense_"+str(architecture_add_on)
            os.mkdir(experiment_path, 0o666)
            for model_manager in models_maangers:
                model = model_manager.get_model((None, timestamp), 31, 2*2*128, rnn_layer_mode, architecture_add_on)
                model_path = experiment_path + "/" + model_manager.get_name()
                os.mkdir(model_path, 0o666)
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                model.summary()
                model.compile(optimizer='adam',
                            loss=loss_fn,
                            metrics=['accuracy'])
                es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
                history = model.fit(x_train, y_train, epochs=15, validation_split=0.2, callbacks=[es])
                conf_matrix_desc_test = visualization.draw_conf_matrix(model, x_test, y_test, labels, "test", model_path)
                conf_matrix_desc_train = visualization.draw_conf_matrix(model, x_train, y_train, labels, "train", model_path)
                hist_description = "train loss, acc: " + str(model.evaluate(x_train, y_train)) + " test loss, acc: " + str(model.evaluate(x_test, y_test))
                acc_description = visualization.get_accuracy_description(model, x_test, y_test, labels)
                visualization.save_txt(str(get_model_summary(model)) + "\n" + str(description) + "\n" + hist_description + "\n" + str(conf_matrix_desc_test) + "\n" + str(conf_matrix_desc_train) + "\n" + str(acc_description) + '\n', model_path)
                visualization.draw_hist(history, model_path)

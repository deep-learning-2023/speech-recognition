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
import preprocessing
import remove_silence
import visualization
import models.simplernn as simpleRNN
import models.GRU as GRU
import models.LSTM as LSTM
x, y, labels= preprocessing.get_raw_data('C:\\Uczymy sie\\sem_11\\data\\train\\audio\\', 4)
x, y = preprocessing.get_mfcc(x, y)
timestamp = x.shape[-1]

print("TensorFlow version:", tf.__version__)
visualization.print_data_description(x, y)
model = GRU.get_model((None, timestamp), 31)
print("timestamp: ", timestamp)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss=loss_fn,
    metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
history = model.fit(x, y, batch_size=2, epochs=5, validation_split=0.2, callbacks=[es])

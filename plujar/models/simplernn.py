import tensorflow as tf
import keras

def get_name():
    return "simpleRNN"

def get_model(_input_shape = (None, 1), classes = 10, neurons = 128, _return_sequences = False, neurons_sizes = None):
    baseModel = keras.models.Sequential([
        keras.Input(shape=_input_shape),
        keras.layers.SimpleRNN(neurons, return_sequences = _return_sequences, activation='relu'),
    ])
    headModel = baseModel.output
    if neurons_sizes == None:
        headModel = tf.keras.layers.Dense(classes, activation="softmax")(headModel)
    elif _return_sequences == False:
        for size in neurons_sizes:
            headModel = tf.keras.layers.Dense(size, activation="relu")(headModel)
        headModel = tf.keras.layers.Dense(classes, activation="softmax")(headModel)
    else:
        for size in neurons_sizes:
            headModel = tf.keras.layers.SimpleRNN(size, activation='relu')(headModel)
        headModel = tf.keras.layers.Dense(classes, activation="softmax")(headModel)
    return keras.Model(inputs=baseModel.input, outputs=headModel)

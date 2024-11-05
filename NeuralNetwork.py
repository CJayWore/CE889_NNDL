import pandas as pd

#Load CSV files for iris datasets
inputs_train=pd.read_csv('normalized_data.csv',usecols = [0,1],skiprows = None,header=None).values
labels_train = pd.read_csv('normalized_data.csv',usecols = [2,3],skiprows = None ,header=None).values.reshape(-1)
inputs_val=pd.read_csv('normalized_data.csv',usecols = [0,1],skiprows = None,header=None).values
labels_val = pd.read_csv('normalized_data.csv',usecols = [2,3],skiprows = None ,header=None).values.reshape(-1)
print("Data loaded (shapes only)", inputs_train.shape, labels_train.shape, inputs_val.shape, labels_val.shape)
# Data loaded

import tensorflow as tf
import keras
from keras import layers

num_inputs=2
num_outputs=2

model = keras.Sequential(name="my_neural_network")
layer0 = layers.Input(shape=(2,)) # input_layer
layer1 = layers.Dense(8, activation="sigmoid") # hidden layer
layer2 = layers.Dense(2, activation="sigmoid")
model.add(layer0)
model.add(layer1)
model.add(layer2)

# model.summary()
model.compile(
    optimizer="SGD",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy'] # This allows the training process to keep track of how many flowers are being classified correctly.
)

history = model.fit(
    inputs_train,
    labels_train,
    batch_size=64,
    epochs=120,
    validation_data=(inputs_val, labels_val),
)

keras.models.save_model(model, "MyModel.keras")

import matplotlib.pyplot as plt
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Set Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
'''
TODO
1. Create a Neural Network:
    2 inputs
    2 outputs
    one hidden layer
2. Tune the network
    The learning rate
    The momentum
    Number neurons in the hidden layer
'''
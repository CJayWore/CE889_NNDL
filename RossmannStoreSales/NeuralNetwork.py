import pandas as pd

#Load CSV
inputs_train=pd.read_csv('DataSet/preprocessed_train.csv',usecols = [0,19],skiprows = 1,header=None).values
outputs_train = pd.read_csv('DataSet/train.csv',usecols = [20] ,header=None).values

inputs_val=pd.read_csv('DataSet/train.csv',usecols = [0,1],skiprows = 1,header=None).values
labels_val = pd.read_csv('DataSet/train.csv',usecols = [2,3],skiprows = 1 ,header=None).values
print("Data loaded (shapes only)", inputs_train.shape, outputs_train.shape, inputs_val.shape, labels_val.shape)
# Data loaded

import tensorflow as tf
import keras
from keras import layers

num_inputs=20
num_outputs=2

model = keras.Sequential(name="RossmannStoreSales")
layer0 = layers.Input(shape=(20,)) # input_layer
layer1 = layers.Dense(6, activation="tanh") # hidden layer
layer2 = layers.Dense(2, activation="softmax")
model.add(layer0)
model.add(layer1)
model.add(layer2)

# model.summary()
model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=['accuracy'] # This allows the training process to keep track of how many flowers are being classified correctly.
)

history = model.fit(
    inputs_train,
    labels_train,
    batch_size=64,
    epochs=1000,
    validation_data=(inputs_val, labels_val),
)

TestInput=pd.read_csv('DataSet/test.csv',usecols = [0,1],skiprows = None,header=None).values
model.predict(TestInput)

keras.models.save_model(model, "RossmannStoreSales.keras")

import matplotlib.pyplot as plt
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Set Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

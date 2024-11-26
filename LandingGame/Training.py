import numpy as np
import pandas as pd
import Neuron

# 读取CSV文件
# data = pd.read_csv('DataSet/trainging.csv')


inputs_train=pd.read_csv('DataSet/raw_data.csv',usecols = [0,1],header=None).values
outputs_train = pd.read_csv('DataSet/raw_data.csv',usecols = [2,3] ,header=None).values

print(inputs_train)
print(outputs_train)
print("Data loaded (shapes only)", inputs_train.shape, outputs_train.shape)





neuralnetwork = Neuron.NeuralNetwork(num_neuron = 3, eta = 0.1, momentum = 0.9)
epochs = 1000
for epochs in range(epochs):
    for input, output in zip(inputs_train, outputs_train):
        predict_outputs = neuralnetwork.feedforward(input)
        neuralnetwork.backpropagate(input,output)

    if epochs % 100 ==0:
        total_error = 0
        for input, output in zip(inputs_train, outputs_train):
            predict_outputs = neuralnetwork.feedforward(input)
            total_error += np.sum((output - predict_outputs)**2)
        print(f"epoch: {epochs}, total error: {total_error}")

for input in inputs_train:
    output = neuralnetwork.feedforward(input)
    print(f"Input: {input}, Output: {output}")





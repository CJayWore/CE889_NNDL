import numpy as np
import pandas as pd
import MyNeuralNetwork as MyNN

# 读取CSV文件
# data = pd.read_csv('DataSet/trainging.csv')

train_data = pd.read_csv('DataSet/train.csv', header=None).values
test_data = pd.read_csv('DataSet/test.csv', header=None).values

'''
column 1 = distance_x
column 2 = distance_y
column 3 = velocity_x
cloumn 4 = velocity_y
'''

inputs_train = train_data[:, :2]
outputs_train = train_data[:, 2:]

inputs_test = test_data[:, :2]
outputs_test = test_data[:, 2:]

print(inputs_train)
print(outputs_train)
print("Data loaded (shapes only)", inputs_train.shape, outputs_train.shape)


network = MyNN.NeuralNetwork(num_neuron=3, eta=0.3, momentum=0.55)

# 训练网络
epochs = 1000
# for epoch in range(epochs):
#     total_loss = 0
#     for i in range(len(inputs_train)):
#         # 前向传播
#         hidden_activations, output_activations = network.feedforward(inputs_train[i])
#         # network.backpropagate(inputs_train[i], outputs_train[i], hidden_activations)
#         # 计算损失
#         error = outputs_train[i] - output_activations
#         total_loss += np.mean(error ** 2)
#
#         # 反向传播
#         network.backpropagate(inputs_train[i], outputs_train[i])
#
#     # 打印每隔 1000 次迭代的损失
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


min_loss_threshold = 1e-5
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(inputs_train)):
        hidden_activations, output_activations = network.feedforward(inputs_train[i])
        error = outputs_train[i] - output_activations
        total_loss += np.mean(error ** 2)
        network.backpropagate(inputs_train[i], outputs_train[i])

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    # 早停条件
    if total_loss < min_loss_threshold:
        print(f"Early stopping at epoch {epoch}, Loss: {total_loss:.4f}")
        break
print("Training complete")

# 测试网络
print("Testing network")
total_test_loss = 0
for i in range(len(inputs_test)):
    _, output_activations = network.feedforward(inputs_test[i])
    error = outputs_test[i] - output_activations
    total_test_loss += np.mean(error ** 2)
    print(f"Input: {inputs_test[i]}, Predicted Output: {output_activations}, Expected Output: {outputs_test[i]}")

print(f"Mean Test Loss: {total_test_loss / len(inputs_test):.4f}")




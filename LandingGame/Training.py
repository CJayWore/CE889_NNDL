import numpy as np
import pandas as pd
import logging
import NeuralNetwork as NN
from sklearn.model_selection import KFold
# 配置日志
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# 加载训练数据和测试数据
logging.info('Loading training and testing data...')
train_data = pd.read_csv('DataSet/train.csv', header=None).values
test_data = pd.read_csv('DataSet/test.csv', header=None).values
# data = pd.read_csv('DataSet/scaled_data.csv', header=None).values

def save_predictions(inputs, predicted_output, targets, filename="predicted_output.csv"):
    data = {
        "Input_1": inputs[:, 0],
        "Input_2": inputs[:, 1],
        "Predicted_Output_1": predicted_output[:, 0],
        "Predicted_Output_2": predicted_output[:, 1],
        "Expected_Output_1": targets[:, 0],
        "Expected_Output_2": targets[:, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

inputs_train = train_data[:,:2]
outputs_train = train_data[:,2:]
inputs_test = train_data[:,:2]
outputs_test = train_data[:,2:]
logging.info(f"Data loaded (shapes only), {inputs_train.shape},{outputs_train.shape}")

nn = NN.NeuralNetwork(learning_rate=0.55, momentum=0.5)
logging.info('Creating Neural Network with learning rate=0.3, momentum=0.55, min_gradient=1e-5')


# 训练神经网络
epochs = 15
logging.info(f'Starting training for {epochs} epochs')
for epoch in range(epochs):
    total_loss = 0
    total_gradient_hidden = np.zeros(len(nn.hidden_layer))  # 隐藏层梯度
    total_gradient_output = np.zeros(len(nn.output_layer))
    for x, y in zip(inputs_train, outputs_train):
        loss, gradient_hidden, gradient_output = nn.backpropagate(x, y)
        total_loss += loss
        total_gradient_hidden += np.array(gradient_hidden)
        total_gradient_output += np.array(gradient_output)


    logging.info(f"Epoch {epoch}, Loss: {total_loss}, Gradient_Hidden: {total_gradient_hidden}, Gradient_Output: {total_gradient_output}")

logging.info('Training completed')

# 测试神经网络
correct_predictions = 0
total_predictions = len(inputs_test)
predicted_outputs = []  # 存储所有测试样本的预测结果

logging.info('Starting testing phase')

def mean_squared_error(predictions, targets):
    return np.mean((predictions - targets) ** 2)

mse_threshold = 0.05  # 设置一个阈值来判断预测是否准确
logging.info(f'Using MSE threshold of {mse_threshold} for accuracy calculation')

for inputs, expected_output in zip(inputs_test, outputs_test):
    predicted_output = nn.feedforward(inputs)
    predicted_outputs.append(predicted_output)  # 保存预测结果
    mse = mean_squared_error(np.array(predicted_output), expected_output)
    if mse < mse_threshold:
        correct_predictions += 1

# 将预测结果保存到 CSV 文件中
predicted_outputs = np.array(predicted_outputs)
save_predictions(inputs_test, predicted_outputs, outputs_test)

# 计算并打印准确率
accuracy = (correct_predictions / total_predictions) * 100
logging.info(f'Test Accuracy: {accuracy:.2f}%')
# print(f"Test Accuracy: {accuracy:.2f}%")

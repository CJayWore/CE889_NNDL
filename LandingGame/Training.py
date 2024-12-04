import numpy as np
import pandas as pd
import logging
import NeuralNetwork as NN



logging.basicConfig(filename='Predictions/TraningLog.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def save_predictions(inputs, predicted_output, targets, filename):
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

# Setting NeuralNetwork
train_normalized = pd.read_csv('DataSet/train_normalized.csv', header=0).to_numpy()

test_normalized = pd.read_csv('DataSet/test_normalized.csv', header=0).to_numpy()
nn = NN.NeuralNetwork(learning_rate=0.00505, momentum=0.55)

logging.info(f'TrainingData_shape: {train_normalized.shape}, TestingData_shape: {test_normalized.shape}')
logging.info(f'Network: learning rate={nn.learning_rate}, momentum={nn.momentum}')

# training
epochs = 100
logging.info('Training in process')


inputs_train = train_normalized[:,:2]
inputs_test = test_normalized[:,:2]
outputs_train = train_normalized[:,2:]
outputs_test = test_normalized[:,2:]

for epoch in range(epochs):
    total_loss = 0
    total_gradient_hidden = np.zeros(len(nn.hidden_layer))  # 隐藏层梯度
    total_gradient_output = np.zeros(len(nn.output_layer))
    for x, y in zip(inputs_train, outputs_train):
        loss, gradient_hidden, gradient_output = nn.backpropagate(x, y)
        total_loss += loss
        total_gradient_hidden += np.array(gradient_hidden)
        total_gradient_output += np.array(gradient_output)


    logging.info(f'     Epoch {epoch+1}, Loss: {total_loss}, Gradient_Hidden: {total_gradient_hidden}, Gradient_Output: {total_gradient_output}')

# Testing
correct_predictions = 0
total_predictions = len(inputs_test)
predicted_outputs = []  # 存储所有测试样本的预测结果

logging.info('Testing in process')




sum_se = 0
for inputs, real_output in zip(inputs_test, outputs_test):
    predicted_output = nn.feedforward(inputs)
    predicted_outputs.append(predicted_output)

    # print(f'RMSE: {rmse}')

# TODO 计算RMSE
predicted_outputs = np.array(predicted_outputs)
rmse = np.sqrt(np.mean((predicted_outputs - outputs_test) ** 2))
logging.info(f'RMSE: {rmse}')

# accuracy = (correct_predictions / total_predictions) * 100
# logging.info(f'Test Accuracy: {accuracy:.2f}%')

save_predictions(
    inputs_test , predicted_outputs, outputs_test, filename=f"Predictions/prediction_RMSE_{rmse:.4f}.csv"
)

logging.info('Cross-validation finished\n\n')
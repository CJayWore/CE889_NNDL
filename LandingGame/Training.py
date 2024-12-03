import numpy as np
import pandas as pd
import logging
import NeuralNetwork as NN
from sklearn.model_selection import KFold



# 配置日志
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
data = pd.read_csv('DataSet/normalized_data.csv', header=None).to_numpy()
nn = NN.NeuralNetwork(learning_rate=0.01, momentum=0.55)
logging.info(f'Data_shape: {data.shape}')
logging.info(f'Network: learning rate={nn.learning_rate}, momentum={nn.momentum}, min_gradient={nn.min_gradient}')

kf = KFold(n_splits=5, random_state=3, shuffle=True)
fold = 0

# training
epochs = 100
logging.info('Training in process')
logging.info(f'n_splits={kf.n_splits}, random_state={kf.random_state}')
for i_train, i_test in kf.split(data):
    fold += 1
    inputs_train = data[i_train,:2]
    inputs_test = data[i_test,:2]
    outputs_train = data[i_train,2:]
    outputs_test = data[i_test,2:]
    logging.info(f'TrainingSet_shape: {inputs_train.shape}, TestingSet_shaape: {inputs_test.shape}')
    logging.info(f'Fold: {fold}')

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

    def root_mean_squared_error(predicted_output, real_output):
        return np.mean((predicted_output - real_output) ** 2)

    rmse_threshold = 0.05  # 设置一个阈值来判断预测是否准确
    logging.info(f'RMSE threshold: {rmse_threshold}')

    for inputs, real_output in zip(inputs_test, outputs_test):
        predicted_output = nn.feedforward(inputs)
        predicted_outputs.append(predicted_output)
        rmse = root_mean_squared_error(np.array(predicted_output), real_output)
        # print(f'RMSE: {rmse}')
        if rmse < rmse_threshold:
            correct_predictions += 1
    # TODO 计算RMSE并且log
    predicted_outputs = np.array(predicted_outputs)
    accuracy = (correct_predictions / total_predictions) * 100
    logging.info(f'Test Accuracy: {accuracy:.2f}%')
    logging.info(f'Fold {fold} completed \n')
    save_predictions(
        inputs_test, predicted_outputs, outputs_test, filename=f"Predictions/prediction_{accuracy:.2f}_Fold_{fold}%.csv"
    )

logging.info('Cross-validation finished\n\n')
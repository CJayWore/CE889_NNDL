import numpy as np
import NeuralNetwork as MyNN


# TODO Test your neural network by running the most recent weights
class NeuralNetHolder:
    def __init__(self, learning_rate = 0.00505, momentum = 0.55):
        super().__init__()
        self.neural_network = MyNN.NeuralNetwork(learning_rate=learning_rate, momentum=momentum)
        # distance_x_min = -813.9287676
        # distance_y_min = 65.50759614
        #
        # distance_x_max = 812.1102333272916
        # distance_y_max = 1023.9753606086138
        #
        # velocity_x_min = -7.083290281
        # velocity_y_min = -7.991966048
        #
        # velocity_x_max = 7.999999999999989
        # velocity_y_max = 7.982954972793153
        self.input_min = np.array([-813.9287676, 65.50759614])  # distance_x_min, distance_y_min
        self.input_max = np.array([812.1102333272916, 1023.9753606086138])  # distance_x_max, distance_y_max

        self.output_min = np.array([-7.083290281, -7.991966048])  # velocity_x_min, velocity_y_min
        self.output_max = np.array([7.999999999999989, 7.982954972793153])  # velocity_x_max, velocity_y_max

    # def normalize_input(self, input_data):
    #     normalized_data = 2 * (input_data - self.input_min) / (self.input_max - self.input_min) - 1
    #     return np.clip(normalized_data, -1, 1)
    #
    # def denormalize_output(self, output_data):
    #     output_data = np.array(output_data,dtype=np.float64)
    #     denormalized_data = (output_data+1) * (self.output_max - self.output_min)/2 + self.output_min
    #     return denormalized_data

    def normalize_input(self, input_data):
        # 计算均值和标准差
        mean = (self.input_max + self.input_min) / 2
        std = (self.input_max - self.input_min) / 2
        # Z-score 标准化
        normalized_data = (input_data - mean) / std
        return normalized_data

    def denormalize_output(self, output_data):
        # 计算均值和标准差
        output_data = np.array(output_data, dtype=np.float64)
        mean = (self.output_max + self.output_min) / 2
        std = (self.output_max - self.output_min) / 2
        # Z-score 反标准化
        denormalized_data = output_data * std + mean
        return denormalized_data

    def process_input(self, input_row):
        # 处理字符串格式的输入
        if isinstance(input_row, str):
            input_row = [float(val.strip()) for val in input_row.split(',')]
        return np.array(input_row, dtype=np.float64)

    def predict(self, input_row):

        print('input_row',input_row)
        input_row=self.process_input(input_row)
        print('processed input_row',input_row)
        normalized_input = self.normalize_input(input_row)
        print('normalized_input',normalized_input)
        predicted_output = self.neural_network.feedforward(normalized_input)
        denormalized_output = self.denormalize_output(predicted_output)
        print('denormalized_output',denormalized_output)
        # VEL_X, VEL_Y = denormalized_output
        # print('VEL_X',VEL_X,'VEL_Y',VEL_Y)

        return denormalized_output

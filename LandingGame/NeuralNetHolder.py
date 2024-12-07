import numpy as np
import NeuralNetwork as MyNN


# TODO Test your neural network by running the most recent weights
class NeuralNetHolder:
    def __init__(self, learning_rate = 0.01, momentum = 0.1):
        super().__init__()
        self.neural_network = MyNN.NeuralNetwork(learning_rate=learning_rate, momentum=momentum)
        self.neural_network.load_weights('best_weights.csv')
        # Distance_X;   Distance_Y;     VEL_X;      VEL_Y
        MIN=np.array([-813.7402115385238, 65.51007146585391, -7.665462826728868, -6.841827957916776])
        MAX=np.array([806.3026531690357, 1025.4204598915912, 7.9999999999999805, 6.557437589810967])
        self.input_min = MIN[:2]  # distance_x_min, distance_y_min
        self.input_max = MAX[:2]  # distance_x_max, distance_y_max 729.4015358007854

        self.output_min = MIN[2:]  # velocity_x_min, velocity_y_min
        self.output_max = MAX[2:]  # velocity_x_max, velocity_y_max

    def normalize_input(self, input_data):
        signs = np.sign(input_data)
        normalized_data = (input_data - self.input_min) / (self.input_max - self.input_min)
        return normalized_data, signs

    def denormalize_output(self, output_data, signs):
        output_data = np.array(output_data,dtype=np.float64)
        denormalized_data = (output_data * (self.output_max - self.output_min) + self.output_min)
        return denormalized_data

    # def normalize_input(self, input_data):
    #     # 计算均值和标准差
    #     mean = (self.input_max + self.input_min)/ 2
    #     std = (self.input_max - self.input_min) / 2
    #     # Z-score
    #     normalized_data = (input_data - mean) / std
    #     return normalized_data
    #
    # def denormalize_output(self, output_data):
    #     # 计算均值和标准差
    #     output_data = np.array(output_data, dtype=np.float64)
    #     mean = (self.output_max + self.output_min) / 2
    #     std = (self.output_max - self.output_min) / 2
    #     # Z-score
    #     denormalized_data = output_data * std + mean
    #     return denormalized_data

    def process_input(self, input_row):
        # 处理字符串格式的输入
        if isinstance(input_row, str):
            input_row = [float(val.strip()) for val in input_row.split(',')]
        return np.array(input_row, dtype=np.float64)

    def predict(self, input_row):

        print('input_row',input_row)
        input_row=self.process_input(input_row)
        print('processed input_row',input_row)

        normalized_input, signs = self.normalize_input(input_row)
        print('normalized_input',normalized_input)
        print('signs',signs)

        predicted_output = self.neural_network.feedforward(normalized_input)
        denormalized_output = self.denormalize_output(predicted_output, signs)
        print('denormalized_output',denormalized_output)

        VEL_X, VEL_Y = denormalized_output
        # print('VEL_X',VEL_X, 'VEL_Y',VEL_Y)
        # print((VEL_X, VEL_Y))
        return denormalized_output

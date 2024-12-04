import numpy as np
import NeuralNetwork as MyNN
from Vector import Vector
import Lander



class NeuralNetHolder:
    def __init__(self, learning_rate = 0.01, momentum = 0.3):
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

    def normalize_input(self, input_data):
        return (input_data - self.input_min) / (self.input_max - self.input_min)

    def denormalize_output(self, output_data):
        return output_data * (self.output_max - self.output_min) + self.output_min

    def get_distance(self, lander):
        distance_x = lander.position.x
        distance_y = lander.position.y
        return distance_x, distance_y

    def controller(self, lander, velocity_x, velocity_y):
        lander.velocity = Vector(velocity_x, velocity_y)

    def prediction(self, lander):
        while True:
            # 1. 获取实时游戏数据 (distance_x, distance_y)
            distance_x, distance_y = self.get_distance(lander)

            # 2. 将输入数据组织成数组并归一化
            input_row = np.array([distance_x, distance_y])
            normalized_input = self.normalize_input(input_row)

            # 3. 使用神经网络进行预测
            predicted_output = self.neural_network.feedforward(normalized_input)
            denormalized_output = self.denormalize_output(predicted_output)

            # 4. 将预测的速度应用到游戏中
            velocity_x, velocity_y = denormalized_output
            self.controller(lander, velocity_x, velocity_y)

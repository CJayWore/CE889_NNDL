import numpy as np


class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.output = 0
        self.input = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def activate(self, inputs):
        inputs = np.array(inputs)

        self.input = inputs
        z = np.dot(self.weights, inputs) + self.bias
        self.output = self.sigmoid(z)
        return self.output

    def loss_function(self, target_output):
        # mean squared error
        return 0.5 * np.square(target_output - self.output)

    def update_weights(self, delta, learning_rate, momentum, previous_delta_weights):
        self.input = np.array(self.input)

        gradient = delta * self.sigmoid_derivative(self.output)
        delta_weights = learning_rate * gradient * self.input + momentum * previous_delta_weights
        self.weights += delta_weights
        self.bias += learning_rate * gradient
        return delta_weights, gradient


class NeuralNetwork:
    def __init__(self, learning_rate=0.3, momentum=0.55, min_gradient=1e-5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.min_gradient = min_gradient

        # 网络结构
        # 输入层：2个神经元，隐藏层：1层3个神经元，输出层：2个神经元
        self.input_layer_size = 2
        self.hidden_layer = [Neuron(self.input_layer_size) for _ in range(3)]
        self.output_layer = [Neuron(len(self.hidden_layer)) for _ in range(2)]
        self.previous_hidden_deltas = [np.zeros(self.input_layer_size) for _ in range(3)]
        self.previous_output_deltas = [np.zeros(len(self.hidden_layer)) for _ in range(2)]

    def feedforward(self, inputs):
        # 隐藏层
        hidden_outputs = [neuron.activate(inputs) for neuron in self.hidden_layer]
        # 输出层
        output_outputs = [neuron.activate(hidden_outputs) for neuron in self.output_layer]
        return output_outputs

    def backpropagate(self, inputs, targets):
        # 前向传播
        hidden_outputs = [neuron.activate(inputs) for neuron in self.hidden_layer]
        output_outputs = [neuron.activate(hidden_outputs) for neuron in self.output_layer]

        # 计算输出层误差
        output_errors = [target - output for target, output in zip(targets, output_outputs)]

        # 更新输出层权重
        gradients_output = []
        for i, neuron in enumerate(self.output_layer):
            delta = output_errors[i]
            self.previous_output_deltas[i], gradient = neuron.update_weights(delta, self.learning_rate, self.momentum,
                                                                             self.previous_output_deltas[i])
            gradients_output.append(gradient)

        # 计算隐藏层误差
        hidden_errors = [
            sum(output_neuron.weights[i] * output_errors[j] for j, output_neuron in enumerate(self.output_layer))
            for i in range(len(self.hidden_layer))
        ]

        # 更新隐藏层权重
        gradients_hidden = []
        for i, neuron in enumerate(self.hidden_layer):
            delta = hidden_errors[i]
            self.previous_hidden_deltas[i], gradient = neuron.update_weights(delta, self.learning_rate, self.momentum,
                                                                             self.previous_hidden_deltas[i])
            gradients_hidden.append(gradient)

        # 计算总误差
        total_loss = sum(neuron.loss_function(target) for neuron, target in zip(self.output_layer, targets))
        return total_loss, gradients_hidden, gradients_output

    # def train(self, inputs, targets, epochs):
    #     for epoch in range(epochs):
    #         total_loss = 0
    #         all_gradients_hidden = []
    #         all_gradients_output = []
    #         for x, y in zip(inputs, targets):
    #             loss, gradients_hidden, gradients_output = self.backpropagate(x, y)
    #             total_loss += loss
    #             all_gradients_hidden.append(gradients_hidden)
    #             all_gradients_output.append(gradients_output)
    #
    #         # 打印损失和梯度
    #         if epoch % 10 == 0:
    #             avg_gradient_hidden = np.mean(all_gradients_hidden, axis=0)
    #             avg_gradient_output = np.mean(all_gradients_output, axis=0)
    #             print(f"Epoch {epoch}, Loss: {total_loss}, Avg Gradient Hidden: {avg_gradient_hidden}, Avg Gradient Output: {avg_gradient_output}")
    #
    #         # 如果梯度太小，则停止训练
    #         if total_loss < self.min_gradient:
    #             break
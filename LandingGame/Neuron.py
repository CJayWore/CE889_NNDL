import numpy as np
import gradient_descent as gd

class Neuron:
    def __init__(self, size_input):
        self.activation_value = 0
        self.weights = np.random.rand(size_input)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def activate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        self.activation_value = self.sigmoid(weighted_sum)
        return self.activation_value

    def loss_function(self, weights, inputs, error):
        self.weights = weights
        activation = self.activate(inputs)
        return np.mean((error - activation)**2) # mean square error

    def update_weights(self, inputs, eta, error, num_steps=1, momentum=0.9, previous_update=None):
        delta_weights = gd.apply_gradient_descent(
            self.weights,
            lambda w: self.loss_function(w, inputs, error),
            eta,
            num_steps,
        )
        if previous_update is not None:
            delta_weights += momentum * previous_update
        self.weights = delta_weights
        return delta_weights  # 返回更新值，方便保存到 previous_update


class NeuralNetwork:
    def __init__(self, num_neuron, eta = 0.1, momentum = 0.9):
        """
        :param num_neuron: hidden layer size
        :param eta: learning rate
        :param momentum: gradient descent momentum
        """
        self.input_layer = [Neuron(2) for _ in range(2)]
        self.hidden_layer = [Neuron(2) for _ in range(num_neuron)]
        self.output_layer = [Neuron(num_neuron) for _ in range(2)]
        self.learning_rate = eta
        self.momentum = momentum
        self.previous_update = [np.zeros_like(neuron.weights) for neuron in self.output_layer]

    def feedforward(self, inputs):
        hidden_activations = np.array([neuron.activate(inputs) for neuron in self.hidden_layer])
        output_activations = np.array([neuron.activate(hidden_activations) for neuron in self.output_layer])
        return output_activations

    def backpropagate(self, inputs, outputs):
        hidden_activations = np.array([neuron.activate(inputs) for neuron in self.hidden_layer])
        output_activations = np.array([neuron.activate(hidden_activations) for neuron in self.output_layer])
        output_errors = outputs - output_activations

        #同时获取索引和值
        for index, neuron in enumerate(self.output_layer):
            previous_update = self.previous_update[index]
            neuron.update_weights(
                hidden_activations,
                self.learning_rate,
                output_errors[index],
                num_steps = 1,
                momentum = self.momentum,
                previous_update = previous_update
            )

        hidden_errors = np.zeros(len(self.hidden_layer))
        for i, hidden_neuron in enumerate(self.hidden_layer):
            hidden_errors[i]=0
            for j in range(len(self.output_layer)):
                hidden_errors[i] += output_errors[j] * self.output_layer[j].weights[i]

            hidden_neuron.update_weights(
                inputs,
                self.learning_rate,
                hidden_errors[i],
                num_steps = 1,
            )

    # def update_weights(self, inputs, eta, error, num_steps = 1):
    #     delta_weights = gd.apply_gradient_descent(
    #         self.weights,
    #         lambda w: self.loss_function(w, inputs, error),
    #         eta,
    #         num_steps,
    #     )
    #     self.weights = delta_weights
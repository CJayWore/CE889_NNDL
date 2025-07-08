import numpy as np
import pandas as pd


class Neuron:
    def __init__(self, input_size, alpha=0.325):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.output = 0
        self.input = None
        self.alpha = alpha

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x*self.alpha))

    def sigmoid_derivative(self, x):
        return self.alpha * x * (1 - x)

    def activate(self, inputs):
        inputs = np.array(inputs)

        self.input = inputs
        z = np.dot(self.weights, inputs) + self.bias
        self.output = self.sigmoid(z)
        return self.output

    def loss_function(self, target_output):
        # mean squared error
        # 0.5 can simplify the loss_function derivative
        return 0.5 * np.square(target_output - self.output)

    def update_weights(self, delta, learning_rate, momentum, previous_delta_weights):
        self.input = np.array(self.input)

        gradient = delta * self.sigmoid_derivative(self.output)
        delta_weights = learning_rate * gradient * self.input + momentum * previous_delta_weights
        self.weights += delta_weights
        self.bias += learning_rate * gradient
        return delta_weights, gradient


class NeuralNetwork:
    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        # self.min_gradient = min_gradient

        # 输入层：2个神经元，隐藏层：1层4个神经元，输出层：2个神经元
        self.input_layer_size = 2
        self.hidden_layer = [Neuron(self.input_layer_size) for _ in range(4)]
        self.output_layer = [Neuron(len(self.hidden_layer)) for _ in range(2)]
        self.previous_hidden_deltas = [np.zeros(self.input_layer_size) for _ in range(4)]
        self.previous_output_deltas = [np.zeros(len(self.hidden_layer)) for _ in range(2)]

    def save_weights(self, filename='best_weights.csv'):
        data = []

        for neuron in self.hidden_layer:
            data.append(['hidden']+neuron.weights.tolist()+[neuron.bias])

        for neuron in self.output_layer:
            data.append(['output'] + neuron.weights.tolist() + [neuron.bias])
        df = pd.DataFrame(data)
        df.to_csv(filename,index=False,header=False)

    def load_weights(self, filename='best_weights.csv'):
        df = pd.read_csv(filename,header=None)
        hidden_index, output_index = 0,0

        for _, row in df.iterrows():
            layer_type = row[0]
            if layer_type == 'hidden':
                bias = row[3]
                weights = np.array(row[1:3], dtype=float)
                self.hidden_layer[hidden_index].weights = weights
                self.hidden_layer[hidden_index].bias = bias
                hidden_index += 1
            elif layer_type == 'output':
                bias = row[5]
                weights = np.array(row[1:5], dtype=float)
                self.output_layer[output_index].weights = weights
                self.output_layer[output_index].bias = bias
                output_index += 1


    def feedforward(self, inputs):
        hidden_outputs = [neuron.activate(inputs) for neuron in self.hidden_layer]
        output_outputs = [neuron.activate(hidden_outputs) for neuron in self.output_layer]
        return output_outputs

    def backpropagate(self, inputs, targets):
        hidden_outputs = [neuron.activate(inputs) for neuron in self.hidden_layer]
        output_outputs = [neuron.activate(hidden_outputs) for neuron in self.output_layer]

        output_errors = [target - output for target, output in zip(targets, output_outputs)]

        # weights update
        gradients_output = []
        for i, neuron in enumerate(self.output_layer):
            delta = output_errors[i]
            self.previous_output_deltas[i], gradient = neuron.update_weights(delta, self.learning_rate, self.momentum,
                                                                             self.previous_output_deltas[i])
            gradients_output.append(gradient)

        hidden_errors = [
            sum(output_neuron.weights[i] * output_errors[j] for j, output_neuron in enumerate(self.output_layer))
            for i in range(len(self.hidden_layer))
        ]

        gradients_hidden = []
        for i, neuron in enumerate(self.hidden_layer):
            delta = hidden_errors[i]
            self.previous_hidden_deltas[i], gradient = neuron.update_weights(delta, self.learning_rate, self.momentum,
                                                                             self.previous_hidden_deltas[i])
            gradients_hidden.append(gradient)

        total_loss = sum(neuron.loss_function(target) for neuron, target in zip(self.output_layer, targets))
        return total_loss, gradients_hidden, gradients_output
import numpy as np
import gradient_descent as gd

class Neuron:
    def __init__(self, size_input):
        self.activation_value = 0
        # self.weights = weight_input
        self.weights = np.random.rand(size_input)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def activate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        self.activation_value = self.sigmoid(weighted_sum)
        return self.activation_value
    
    def weight_multiply(self, weight_index):
        return self.activation_value * self.weights[weight_index]

    def loss_function(self, weights, inputs, error):
        self.weights = weights
        activation = self.activate(inputs)
        return np.sum((error - activation)**2) #square error

    def update_weights(self, inputs, eta, error, num_steps = 1):
        update_weights = gd.apply_gradient_descent(
            self.weights,
            lambda w: self.loss_function(w, inputs, error),
            eta,
            num_steps,
        )
        self.weights = update_weights

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
            )



#Test
inputs = np.array([
    [0,0],[0,1],[1,0],[1,1]
])
outputs = np.array([
    [0],[1],[1],[0]
])

print("测试ffnn")
nn = NeuralNetwork(num_neuron = 3, eta = 0.1, momentum = 0.9)
for input_data in inputs:
    output = nn.feedforward(input_data)
    print(f"Input: {input_data}, Output: {output}")

print(("测试ffnn和bpg"))
epochs = 1000
for epochs in range(epochs):
    for input_data, target_output in zip(inputs, outputs):
        predict_outputs = nn.feedforward(input_data)
        nn.backpropagate(input_data, target_output)

    if epochs % 100 ==0:
        total_error = 0
        for input_data, target_output in zip(inputs, outputs):
            predict_outputs = nn.feedforward(input_data)
            total_error += np.sum((target_output - predict_outputs)**2)
        print(f"epoch {epochs}, total error: {total_error}")

for input_data in inputs:
    output = nn.feedforward(input_data)
    print(f"Input: {input_data}, Output: {output}")
import numpy as np
import gradient_descent as gd
import logging

# logging.basicConfig(filename='MyNeuralNetwork.log', level=logging.DEBUG)
np.random.seed(42)

class Neuron:
    def __init__(self, size_input):
        self.activation_value = 0
        self.weights = np.random.rand(size_input) * np.sqrt(1/size_input)
        self.bias = np.random.rand()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def activate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.activation_value = self.sigmoid(weighted_sum)
        return self.activation_value

    def loss_function(self, weights, inputs, target_outputs):
        original_weights = self.weights.copy()
        self.weights = weights
        activation = self.activate(inputs)
        self.weights = original_weights  # 恢复原来的权重，防止副作用
        return np.mean((target_outputs - activation) ** 2) # mean square error

    # def update_weights(self, inputs, eta, error, num_steps=1, momentum=0.55, previous_update=None):
    #     delta_weights = gd.apply_gradient_descent(
    #         self.weights,
    #         lambda w: self.loss_function(w, inputs, error),
    #         eta,
    #         num_steps,
    #     )
    #     if previous_update is not None:
    #         delta_weights += momentum * previous_update
    #     self.weights -= delta_weights
    #     self.bias -= eta * error  # Update bias term
    #     return delta_weights

    def update_weights(self, inputs, eta, error, momentum=0.55, previous_update=None):
        gradient = error * self.sigmoid_derivative(self.activation_value) * inputs
        delta_weights = eta * gradient
        if previous_update is not None:
            delta_weights += momentum * previous_update

        # logging.debug(f"Gradient: {gradient}")
        # logging.debug(f"Delta Weights: {delta_weights}")
        # logging.debug(f"Previous Weights: {self.weights}")

        self.weights -= delta_weights
        self.bias -= eta * error # Update bias term(sigmoid)

        # logging.debug(f"Updated Weights: {self.weights}")
        # logging.debug(f"Updated Bias: {self.bias}")

        return delta_weights



class NeuralNetwork:
    def __init__(self, num_neuron, eta=0.3, momentum=0.55, min_grad=1e-5):
        """
        :param num_neuron: hidden layer size
        :param eta: learning rate
        :param momentum: gradient descent momentum
        :param min_grad: minimum gradient for stopping criteria
        """
        self.input_layer = [Neuron(2) for _ in range(2)]
        self.hidden_layer = [Neuron(2) for _ in range(num_neuron)]
        self.output_layer = [Neuron(num_neuron) for _ in range(2)]
        self.learning_rate = eta
        self.momentum = momentum
        self.min_grad = min_grad
        self.previous_updates_output = [np.zeros_like(neuron.weights) for neuron in self.output_layer]
        self.previous_updates_hidden = [np.zeros_like(neuron.weights) for neuron in self.hidden_layer]

    def feedforward(self, inputs):
        hidden_activations = np.array([neuron.activate(inputs) for neuron in self.hidden_layer])
        output_activations = np.array([neuron.activate(hidden_activations) for neuron in self.output_layer])
        return hidden_activations, output_activations

    def backpropagate(self, inputs, outputs):
        hidden_activations, output_activations = self.feedforward(inputs)
        output_errors = outputs - output_activations

        # logging.debug(f"Output Errors: {output_errors}")

        # Update output layer weights
        for index, neuron in enumerate(self.output_layer):
            previous_update = self.previous_updates_output[index]
            gradient = output_errors[index] * neuron.sigmoid_derivative(output_activations[index])

            # logging.debug(f"Output Layer {index} Gradient: {gradient}")

            self.previous_updates_output[index] = neuron.update_weights(
                hidden_activations,
                self.learning_rate,
                gradient,
                # num_steps=1,
                momentum=self.momentum,
                previous_update=previous_update
            )
        # Calculate hidden layer errors and update hidden layer weights
        hidden_errors = np.zeros(len(self.hidden_layer))
        for i, hidden_neuron in enumerate(self.hidden_layer):
            hidden_errors[i] = 0
            for j in range(len(self.output_layer)):
                hidden_errors[i] += output_errors[j] * self.output_layer[j].weights[i]

            # Apply sigmoid derivative to the hidden layer error
            hidden_derivative = hidden_neuron.sigmoid_derivative(hidden_activations[i])
            hidden_errors[i] *= hidden_derivative

            # logging.debug(f"Hidden Layer {i} Error: {hidden_errors[i]}")
            # logging.debug(f"Hidden Layer {i} Derivative: {hidden_derivative}")

            previous_update = self.previous_updates_hidden[i]
            self.previous_updates_hidden[i] = hidden_neuron.update_weights(
                inputs,
                self.learning_rate,
                hidden_errors[i],
                # num_steps=1,
                momentum=self.momentum,
                previous_update=previous_update
            )

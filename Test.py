

import unittest
import numpy as np
from  Neuron import Neuron 

class TestNeuron(unittest.TestCase):
    def setUp(self):
        self.neuron = Neuron(0.5, [0.1, 0.2, 0.3])

    def test_sigmoid(self):
        self.assertAlmostEqual(self.neuron.sigmoid(0), 0.5)
        self.assertAlmostEqual(self.neuron.sigmoid(2), 1 / (1 + np.exp(-2)))

    def test_activate(self):
        self.assertAlmostEqual(self.neuron.activate(1), self.neuron.sigmoid(1))

    def test_weight_multiplication(self):
        self.neuron.activate(1)  # 激活神经元
        self.assertAlmostEqual(self.neuron.weight_multiplication(0), self.neuron.activation_value * 0.1)
        self.assertAlmostEqual(self.neuron.weight_multiplication(1), self.neuron.activation_value * 0.2)

    def test_weight_update(self):
        self.neuron.activate(1)
        self.neuron.weight_update(0.1, 0.5)
        self.assertAlmostEqual(self.neuron.weights[0], 0.1 + 0.1 * 0.5 * self.neuron.activation_value)
        self.assertAlmostEqual(self.neuron.weights[1], 0.2 + 0.1 * 0.5 * self.neuron.activation_value)

if __name__ == '__main__':
    unittest.main()


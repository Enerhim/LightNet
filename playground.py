from lib.lightnet import NeuralNetwork
import numpy as np

network = NeuralNetwork()
network.add_layer(784, "sigmoid")
network.add_layer(16, "sigmoid")
network.add_layer(16, "sigmoid")
network.add_layer(10, "sigmoid")
network.init_params()
print(network.feed_forward_once(np.random.randn(1, 784)) )
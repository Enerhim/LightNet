import numpy as np
import math

'''
Reference 
=========
Building a neural network FROM SCRATCH: https://www.youtube.com/watch?v=w8yWXqWQYmU
3B1B Neural Network Series: https://www.youtube.com/watch?v=aircAruvnKk
Sebastian Lague Identify Doodles: https://www.youtube.com/watch?v=hfMk-kjRv4c
=========
'''

'''
Goals
=========
- Able to recognize handwritten digits
- Optimized and fast
- Graphing automatically
- Various activation functions
- Simple and well documented
- Minimum imports
- Splitable train and test dataset functions
- General purpose
- Light weight
- Ease of use
'''

class NeuralNetwork:
    def __init__(self, config:list=[]) -> None:
        # Configuration for neural network
        self.config = config
        self.act_functions = []
        
        # Initiate random weights and biases
    def init_params(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.config)):
            m = self.config[i]
            
            # Check if at last layer to generate K
            try: k = self.config[i+1]
            except IndexError: break
            
            # Add weights for entire layer to weights of network
            # Weights of layer = Matrix(m, k)
            # m is the input size and k is the output size
            self.weights.append(np.random.randn(m, k)) 
            self.biases.append(np.random.randn(1, k))
            
        self.weights = np.array(self.weights)
        # self.biases = np.array(self.biases)
                        
    def activation(self, x):
        return 1 / (1 + math.exp(-x))
            

    def feed_forward_once(self, input_):
        # Input = Matrix(m, 1)
        
        # Set activations to input layer initially
        Z = input_
        # Vectorize the activation function
        act_vec = np.vectorize(self.activation)
        # For each layer matrix 
        for i, _ in enumerate(self.weights):
            # Take the dot product of that layer matrix with input vector and add bias
            Z = act_vec(np.dot(Z, self.weights[i]) + self.biases[i])
        
        return Z
            
    
    def add_layer(self, no_neurons:int, activation:str):
        # Add length of layer to config
        self.config.append(no_neurons)
        # Add the activation function for that specific layer 
        self.act_functions.append(activation)    
        
network = NeuralNetwork()
network.add_layer(784, "sigmoid")
network.add_layer(16, "sigmoid")
network.add_layer(16, "sigmoid")
network.add_layer(10, "sigmoid")
network.init_params()
print(network.feed_forward_once(np.random.randn(1, 784)) )
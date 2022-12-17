import numpy as np

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
        
    def init_params(self):
        # Initiate random weights and biases
        self.weights = np.array()
        self.biases = np.array()
        
    
    def feed_forward(self):
        pass
        
    def add_layer(self, no_neurons:int, activation:str):
        # Add length of layer to config
        self.config.append(no_neurons)
        # Add the activation function for that specific layer 
        self.act_functions.append(activation)    
    
network = NeuralNetwork()
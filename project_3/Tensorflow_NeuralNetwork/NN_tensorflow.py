import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class NeuralNetwork(tf.keras.Sequential):
    """ Class for Tensorflow based Neural Network,
        using Adam for optimization and MSE as loss function.
        
        args:
            layers                 (int)  : Number of hidden layers 
            input_size             (tuple): Shape of input layer 
            learning rate          (int)  : Learning rate
        """
    def __init__(self, layers, input_size, learning_rate=0.001):
        super(NeuralNetwork,self).__init__()

        # Creating the first layers, the input layer. 
        self.add(tf.keras.layers.Dense(layers[0],input_shape=(input_size,),activation= None))
        
        # Creating hidden layers
        for layer in layers[1:-1]:
            self.add(tf.keras.layers.Dense(layer,activation ='tanh'))

        #Creating the last, input layer.
        self.add(tf.keras.layers.Dense(layers[-1],activation='linear'))
       
        # Creating optimizer 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        
        #Creating loss function MSE
        self.loss_func = tf.keras.losses.MeanSquaredError()
        
    @tf.function
    def comp_gradients(self):
        #computing Gradient
        with tf.GradientTape() as tape:
            loss = self.MSE()
        gradients = tape.gradient(loss,self.trainable_variables)
        return loss, gradients
        
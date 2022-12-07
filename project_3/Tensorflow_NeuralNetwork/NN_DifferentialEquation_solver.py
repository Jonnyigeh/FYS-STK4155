import tensorflow as tf
import numpy as np
from NN_tensorflow import NeuralNetwork
from tensorflow.python.ops.numpy_ops import np_config
            # np_config.enable_numpy_behavior()



np.random.seed(2023)

class Neural_DE_solver(NeuralNetwork):
    def __init__(self, layers, input_size, learning_rate):
        super(Neural_DE_solver,self).__init__(layers,input_size,learning_rate)
    


    @tf.function
    def predict(self, x, t):
        """
        args: 

            x, t                       (np.array): datasets
            trial_f                    (np.array): trial solution for  Differential Equation(DE)
        """    
    
        self.x = x
        self.t = t
        trial_f = self.trial_solution(training=False)
        return trial_f


    @tf.function
    def trial_solution(self,training):
        """
        args: 
        
            x, t                       (np.array): datasets
            X                          (np.array): Concatenates x and t values in 1 shape.
            N                          (np.array): Feed Forward Neural Network                
        
        returns:
            trial_f                    (np.array): trial solution for the DE.
        
        """
        x, t = self.x, self.t
        X = tf.concat([x,t],1)
        N = self(X, training= training)
        trial_f = tf.sin(np.pi*x) + t*x*(1-x)*N
        return trial_f

    @tf.function
    def MSE(self):
        """
        args:
            x, t            (np.array): datasets
        
        
        returns:
            loss                 (int): Mean Squared Error
        
        """
        x,t = self.x, self.t
        with tf.GradientTape() as gg:
            gg.watch(x)
            with tf.GradientTape(persistent = True) as g:
                g.watch([x,t])
                trial_f = self.trial_solution(training = True)

            df_dt = g.gradient(trial_f,t)
            df_dx = g.gradient(trial_f,x)
        d2f_dx2 = gg.gradient(df_dx,x)

        y_predict = df_dt - d2f_dx2
        loss = self.loss_func(0., y_predict)
        return loss

    def fit(self,x,t,epochs):
        """
        args: 



        """
        x = x.reshape(-1,1)
        t = t.reshape(-1,1)
        x = tf.convert_to_tensor(x,dtype=tf.float32)
        t = tf.convert_to_tensor(t,dtype=tf.float32)

        self.x = x
        self.t = t

        epochs_ = np.zeros(epochs)
        n_loss = np.zeros(epochs)

    
        for epoch in range(epochs):
            loss, gradients = self.comp_gradients()
            self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
            epochs_[epoch] = epoch 
            n_loss[epoch] = loss

        return n_loss, epochs_

    
    
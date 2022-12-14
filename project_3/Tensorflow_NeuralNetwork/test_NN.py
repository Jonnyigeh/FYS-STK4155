import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from NN_DifferentialEquation_solver import Neural_DE_solver
import seaborn as sns
import pandas as pd


sns.set_theme() 
n = 10
x = np.linspace(0, 1, n)
t = np.linspace(0, 1, n)
x, t = np.meshgrid(x,t)
x, t = x.ravel(),t.ravel()
layers = [100]*4+[1]
input_size = 2
epochs = 900
analytical_solution = lambda x,t: np.sin(x*np.pi)*np.exp(-np.pi**2 * t)


if True: # 0. Setup for calculations:

    ''' 
        0. Setup for calculations: 
            - Sets up and splitting train and test data.
            - Creating model for train data.
            - Setting up environment for use of Tensorflow modules. 

        1. True/False: 
            - Calculating MSE of predict model
            - Comparin MSE of train data and test data.
            - Plotting MSE of train data as function of epochs.
            

        2. True False: 
            - Plotting analytical solution vs Tensorflow approximation of the DE 
                for various time steps, ranging from [t=1,t=2,t=3 and t=8].
    
    ''' 
    # Train test split data using Sklearn module
    X_train, X_test, y_train, y_test = train_test_split(x,t, test_size = 0.2)
    
    model_train = Neural_DE_solver(layers= layers,
                                input_size= input_size,
                                learning_rate=0.001)

    
    n_loss1, epochs_1 = model_train.fit(x=X_train,t = y_train,epochs=epochs)    
    
    # Converting train and test data to tensors of float32 datatype.
    X_train, y_train = tf.cast(X_train, dtype = tf.float32), tf.cast(y_train, dtype = tf.float32)
    X_test, y_test  = tf.cast(X_test, dtype = tf.float32), tf.cast(y_test, dtype = tf.float32)
    
    
    #Creating tensorflow meshgrid with spesific dtype        
    number_grid_points = 30
    
    start_x = tf.constant(0.0, dtype = tf.float32)
    stop_x = tf.constant(1.0, dtype = tf.float32)
    
    start_t = tf.constant(0.0, dtype = tf.float32)
    stop_t = tf.constant(1.0, dtype = tf.float32)

    X, T = tf.meshgrid(tf.linspace(start_x,stop_x,number_grid_points),tf.linspace(start_t,stop_t,number_grid_points))
    x, t = tf.reshape(X, [-1,1]), tf.reshape(T,[-1,1])

    #Make sure that the test data tensors have compatible shapes.
    X_test = tf.expand_dims(X_test, axis=1)
    y_test = tf.expand_dims(y_test, axis=1)
    
    
    #Setting up fit model and predict model
    prediction_model = model_train.predict(x = X_test, t = y_test)



    if True: # 1. True/False
        
        #Calucaltion MSE of predict model
        Squared_errors = tf.math.squared_difference(0., prediction_model)
        mean_squared_error = tf.math.reduce_mean(Squared_errors)
        
        # print the MSE of train and test data
        print('\n\n\n\n\n\n\n')
        print(f"Mean squared error of test data: {mean_squared_error}")
        print(f"Mean squared errof of train data: {n_loss1[-1]}")
        print('\n\n\n\n\n\n\n')
        
        #Plotting MSE of train data over epochs
        plt.plot(epochs_1,n_loss1, label =' MSE of train data')
        plt.title('Hidden layers = 4, Neurons = 100 , Epochs = 900')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()
        # plt.savefig('/figs/MSE_train_data_900epochs.pdf')
        plt.show()

    if False: # 2. True/False
        n = 10
        x = np.linspace(0, 1, n)
        t = np.linspace(0, 1, n)
        x, t = np.meshgrid(x,t)
        x, t = x.ravel(),t.ravel()
        layers = [100]*4+[1]
        input_size = 2
        epochs = 900

        #Creating prediction model
        model = Neural_DE_solver(layers = layers,
                                input_size=  input_size,
                                learning_rate = 0.001)
        n_loss, epochs_ = model.fit(x=x, t = t, epochs=epochs)
        number_grid_points = 30
        start_x = tf.constant(0.0, dtype = tf.float32)
        stop_x = tf.constant(1.0, dtype = tf.float32)
        start_t = tf.constant(0.0, dtype = tf.float32)
        stop_t = tf.constant(1.0, dtype = tf.float32)
        X, T = tf.meshgrid(tf.linspace(start_x,stop_x,number_grid_points),tf.linspace(start_t,stop_t,number_grid_points))
        x, t = tf.reshape(X, [-1,1]), tf.reshape(T,[-1,1])
        
        #Setting up fit model and predict model

        prediction_model = model.predict(x,t)
        
        #Creating arbitrary linspace
        x_plot = np.linspace(0,1,30)

        #Plotting solutions and comparing.
        # Given t = 1
        plt.plot(x_plot,analytical_solution(x_plot,t[0]), color = 'black',label = ' analytical solution')
        plt.plot(x_plot,prediction_model[0:30],color ='red',linestyle='dashed', label= ' Tensorflow approximation')
        
        #Given t = 2
        plt.plot(x_plot,analytical_solution(x_plot,t[35]), color = 'black')
        plt.plot(x_plot,prediction_model[30:60],color ='red',linestyle='dashed')
        
        #Given t = 3
        plt.plot(x_plot,analytical_solution(x_plot,t[70]),color = 'black')
        plt.plot(x_plot,prediction_model[60:90],color ='red',linestyle='dashed')
        
        #Given t = 8
        plt.plot(x_plot,analytical_solution(x_plot,t[260]),color = 'black')
        plt.plot(x_plot,prediction_model[240:270],color ='red',linestyle='dashed')
        
        plt.legend(['Analytical solution', 'Tensorflow approximation'],loc = 'upper right',prop={'size': 8})
        plt.ylabel('u(x,t)')
        plt.xlabel('x')
        # plt.savefig('figs/Analytical_solution_vs_tensorflow.pdf')
        plt.show()

if False: # Create plot of Analytic solution for t => 0, over datapoints 
    analytical_solution = lambda x,t: np.sin(x*np.pi)*np.exp(-np.pi**2 * t)
    model = Neural_DE_solver(layers=layers,input_size=input_size, learning_rate=0.001)
    n = 10
    x = np.linspace(0,1,n)
    t = np.linspace(0,1,n)
    x,t = x.ravel(), t.ravel()

    plt.plot(x,analytical_solution(x,t=0), label = 't = 0')
    plt.plot(x,analytical_solution(x,t =0.05), label='t = 0.05')
    plt.plot(x,analytical_solution(x,t=0.1), label = 't = 0.1')
    plt.plot(x,analytical_solution(x,t=1), label = 't = 1.0')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    # plt.savefig('figs/analytical_solution_t=0.01.pdf')
    plt.show()


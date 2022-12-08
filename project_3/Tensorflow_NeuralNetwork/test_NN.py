import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from NN_DifferentialEquation_solver import Neural_DE_solver
import seaborn as sns
import pandas as pd
sns.set_theme()
n = 1000
x = np.linspace(0, 1, n)
t = np.linspace(0, 1, n)
x, t = np.meshgrid(x,t)
x, t = x.ravel(),t.ravel()
# layers = [100]*4+[1]
input_size = 2
epochs = 1000
analytical_solution = lambda x,t: np.sin(x*np.pi)*np.exp(-np.pi**2 * t)
my_model = Neural_DE_solver(layers=layers, input_sz=input_sz, learning_rate=0.001)
if False: #[UNFINISHED] Trying to solve this problem: Fit the model, later evaluate the function at any t 0 <= x <= 1 and any t > 0:

    my_model1 = Neural_DE_solver(layers=layers, input_size=input_size,learning_rate=0.001)
    my_model2 = Neural_DE_solver(layers=layers, input_size=input_size,learning_rate=0.001)
    n_loss, epochs_ = my_model1.fit(x=x,t=t,epochs=epochs)

    x = tf.convert_to_tensor(x,dtype=tf.float32)
    t = tf.convert_to_tensor(t,dtype=tf.float32)

    x = np.linspace(0, 1, n)
    t = np.linspace(0, 1, n)
    x, t = np.meshgrid(x,t)
    x, t = x.ravel(),t.ravel()
    prediction = my_model2.predict(x=x,t=t)
    plt.plot(np.linspace(0,100,100),prediction)
    plt.show()
if False: #[FINISHED]Create plot of Analytic solution for t > 0, over datapoints.
    analytical_solution = lambda x,t: np.sin(x*np.pi)*np.exp(-np.pi**2 * t)

    n = 10
    x = np.linspace(0,1,n)
    t = np.linspace(0,1,n)
    breakpoint()
    x, t = np.meshgrid(x, t)
    x ,t = x.ravel(), t.ravel()

    plt.plot(np.linspace(0,1,100),analytical_solution(x,t ))
    plt.xlabel('linspace(0,1,100)')
    plt.ylabel('u(x,t)')
    # plt.savefig('figs/PDE_function_over_time.pdf')
    plt.show()
if False: #[FINISHED] Create plot of PDE using loss function MSE

    
    Neuralnetwork_model = Neural_DE_solver(layers=layers,
                                            input_size=input_size,
                                            learning_rate=0.001)
    n_loss, epochs_ = Neuralnetwork_model.fit(x=x,t=t,epochs=epochs)
    plt.plot(epochs_,n_loss,label = 'Optimizer: Adam')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Number of epochs')
    plt.legend()
    plt.show()
if False: #[FINISHED] Create plot of loss functions over epochs = 1000, neurons = 100
    Model1 = Neural_DE_solver(layers=[100]*4+[1],
                                input_size=input_size,
                                learning_rate=0.001)
    Model2 = Neural_DE_solver(layers=[100]*3+[1],
                                input_size=input_size,
                                learning_rate=0.001)
    Model3 = Neural_DE_solver(layers=[100]*2+[1],
                                input_size=input_size,
                                learning_rate=0.001)
    Model4 = Neural_DE_solver(layers=[100]*1+[1],
                                input_size=input_size,
                                learning_rate=0.001)
    n_loss_1, epochs_ = Model1.fit(x=x,t=t,epochs=epochs)
    n_loss_2, epochs_ = Model2.fit(x=x,t=t,epochs=epochs)
    n_loss_3, epochs_ = Model3.fit(x=x,t=t,epochs=epochs)
    n_loss_4, epochs_ = Model4.fit(x=x,t=t,epochs=epochs)
    plt.plot(epochs_,n_loss_1, label =' 4 hidden layers')
    plt.plot(epochs_,n_loss_2, label =' 3 hidden layers')
    plt.plot(epochs_,n_loss_3, label =' 2 hidden layers')
    plt.plot(epochs_,n_loss_4, label =' 1 hidden layers')
    plt.title('Epochs = 1000, Neurons = 100, Activation = tanh')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Number of Epochs')
    plt.legend(loc = 'upper right')
    # plt.savefig('figs/MSE_1000epochs_tanh_nr2.pdf')
    plt.show()  
if False: #[UNFINISHED] Trying to solve this problem: Create heatmap of nodes vs layers to find optimal parameteres for loss/cost function.
                                                                                    
    hidden_layers = np.array((2,4,6,8,10,20,30,50))
    layers_ = np.array((10,20,30,40,50,100,200,300))
    total_loss = np.zeros((len(hidden_layers),len(layers_)))
    model = Neural_DE_solver(layers = hidden_layers, nodes = layers_,
                            input_size = input_size, 
                            learning_rate = 0.001)
    for i, hidden_layers_ in enumerate(hidden_layers):
        for j, _layers_ in enumerate(layers_):
            n_loss = model.fit(x=x,t=t,epochs=epochs)
            total_loss[i][j] = model.MSE()
    df = pd.DataFrame(data=total_loss[:,:],
                        index= hidden_layers,
                         columns= layers_)

    print(df)
    sns.heatmap(df,annot=True,fmt='.2g')
    plt.xlabel("Layers")
    plt.ylabel("Nodes")
    plt.show()


if True: #Testing something
    grid_span = 40
    prediciton = my_model.predict(x,t)
    reshaped_analytical = tf.reshape(analytical_solution(x,t),(grid_span,grid_span))
    reshaped_prediction = tf.reshape(prediciton,(grid_span,grid_span))
    relative_error = np.abs((reshaped_analytical-reshaped_prediction)/reshaped_analytical)
    breakpoint()

            
            





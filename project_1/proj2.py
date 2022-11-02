# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from random import random, seed
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
from sys import exit
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys



# class NeuralNetwork:

#     def __init__(self,x_data,y_data,n_hidden_neurons=50,n_categories=10,epochs=10,batch_size=100,eta=0.1,lmbd=0.0):
#         self.X_data_full = X_data
#         self.Y_data_full = Y_data

#         self.n_inputs = X_data.shape[0]
#         self.n_features = X_data.shape[1]
#         self.n_hidden_neurons = n_hidden_neurons
#         self.n_categories = n_categories

#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.iterations = self.n_inputs // self.batch_size
#         self.eta = eta
#         self.lmbd = lmbd

#         self.create_biases_and_weights()

#     def create_biases_and_weights(self):
#         self.hidden_weights = np.random.randn(self.n_features,self.n_hidden_neurons)
#         self.hidden_bias = no.zeros(self.n_hidden_neurons) + 0.01

    # mnist = tf.keras.datasets.mnist

    # (x_train, y_train),(x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(28, 28)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(10, activation='softmax')])
    # model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    # model.fit(x_train, y_train, epochs=5)
    # model.evaluate(x_test, y_test)

    # def func(x):
    #     x**2+4
    #     return x

    # def create_X(x,n):
    #     if len(x.shape) > 1:
    #         x = np.ravel(x)

    #     N = len(x)
    #     l = int((n + 1) * (n + 2) / 2)
    #     X = np.ones((N,l))

    #     for i in range(1, n+1):
    #         q = int((i) * (i+1) / 2)
    #         for k in range(i+1):
    #             X[:,q+k] = (x **(i - k)) * (y **k)

    #     return X




#     def backpropagation(z):
# # Importing various packages


# the number of datapoints
n = 100
x = 2*np.random.rand(n,1)
y = 4+3*x+np.random.randn(n,1)

X = np.c_[np.ones((n,1)), x]
print(x)
# Hessian matrix
H = (2.0/n)* X.T @ X
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
print(EigValues)

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
print(beta_linreg)
beta = np.random.randn(2,1)

eta = 1.0/np.max(EigValues)
Niterations = 1000

for iter in range(Niterations):
    gradient = (2.0/n)*X.T @ (X @ beta-y)
    beta -= eta*gradient

print(beta)
xnew = np.array([[0],[2]])
xbnew = np.c_[np.ones((2,1)), xnew]
ypredict = xbnew.dot(beta)
ypredict2 = xbnew.dot(beta_linreg)
plt.plot(xnew, ypredict, "r-")
plt.plot(xnew, ypredict2, "b-")
plt.plot(x, y ,'ro')
plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Gradient descent example')
plt.show()





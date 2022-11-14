import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn import datasets
from sys import exit

np.random.seed(0)
class Layer():
    def __init__(
            self,
            dimension_in,
            dimension_out,
            act_func="sigmoid",
            input_layer=False,
            output_layer=False,
            output_dimension="Not defined"):

        self.dimension_in = dimension_in
        self.dimension_out = dimension_out
        self.act_func = act_func
        self.output_layer = output_layer
        if output_layer:
            self.act_func="linear"
            self.dimension_out = 1
        #     try:
        #         self.dimension_out = float(output_dimension)
        #     except ValueError:
        #         print("Please specify output dimension")

        if input_layer:
            self.weights = np.ones((self.dimension_in, self.dimension_out))
            self.bias = np.zeros((1, self.dimension_out))
        else:
            self.weights = np.random.randn(self.dimension_in, self.dimension_out)
            self.bias = np.zeros((1, self.dimension_out)) + 0.01

    def forward_prop(self, input_value):
        self.input = input_value
        self.hidden_val = self.input @ self.weights + self.bias
        self.output = self.activation_function(self.hidden_val)

        return self.output

    def backward_prop(self, error_prev_layer, lmbda):
        if self.output_layer:
            self.grad_w = self.input.T @ error_prev_layer
            self.grad_b = np.sum(error_prev_layer, axis=0)
            if lmbda > 0.0:
                self.grad_w += lmbda * self.weights

            self.error_layer = error_prev_layer @ self.weights.T * self.grad_activation_function(self.output)

            return self.error_layer
        else:
            self.error_layer = error_prev_layer @ self.weights.T * self.grad_activation_function(self.output)
            self.grad_w = self.input.T @ self.error_layer
            self.grad_b = np.sum(self.error_layer, axis=0)
        if lmbda > 0.0:
            self.grad_w += lmbda * self.weights

        # To avoid exploding gradients, we do gradient clipping
        # by removing a factor of 10 when the gradient exceed some threshold
        self.grad_w = np.where(self.grad_w > 10, 0.1 * self.grad_w, self.grad_w)
        self.grad_b[self.grad_b>10] = 0.1

        return self.error_layer

    def activation_function(self, x):
        if self.act_func=="sigmoid":
            return 1 / (1 + np.exp(-x))

        if self.act_func=="linear":
            return x

        if self.act_func=="relu":
            return np.maximum(0,x)

    def grad_activation_function(self, x):
        if self.act_func=="sigmoid":
            return self.activation_function(x) * ( 1 - self.activation_function(x))

        if self.act_func=="linear":
            return np.ones_like(x)

        if self.act_func=="relu":
            return np.where(x>0, 1, 0.01 * x) # Leak ReLu

        # if activation_function="relu":
        #     if x <= 0:
        #         return 0
        #     else:
        #         return 1


class NeuralNetwork():
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_layers=10,
            n_hidden_neurons=50,
            epochs=10,
            batch_size=10,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_outputs = self.n_inputs
        self.n_hidden_layers = n_hidden_layers

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.layers = []
        self.MSE_train = []
        self.initialize_layers()

    def initialize_layers(self):
        """Initializes the network with all hidden layers with dimensions (n_neurons,n_neurons)
            also adds an input layer with dimensions (n_features, n_neurons)
                plus an output layer with dimensions (n_neurons, n_outputs)
        """
        input_layer = Layer(self.n_features, self.n_hidden_neurons)
        self.layers.append(input_layer)
        for l in range(self.n_hidden_layers):
            self.layers.append(Layer(self.n_hidden_neurons, self.n_hidden_neurons))

        output_layer = Layer(self.n_hidden_neurons, self.n_outputs, output_layer=True)
        self.layers.append(output_layer)


    def feed_forward(self):
        input_layer = self.layers[0]
        output_prev = input_layer.forward_prop(self.X_data)
        # Feed through hidden layers
        for l in range(self.n_hidden_layers):
            hidden_layer = self.layers[l+1]
            output = hidden_layer.forward_prop(output_prev)
            output_prev = output
        # And through output layer
        output_layer = self.layers[-1]
        self.actual_output = output_layer.forward_prop(output_prev)


    def feed_forward_out(self, X, Y):
        input_layer = self.layers[0]
        output_prev = input_layer.forward_prop(X)
        # Feed through hidden layers
        for l in range(self.n_hidden_layers):
            hidden_layer = self.layers[l+1]
            output = hidden_layer.forward_prop(output_prev)
            output_prev = output
        # And through output layer
        output_layer = self.layers[-1]
        actual_output = output_layer.forward_prop(output_prev)
        MSE = mean_squared_error(Y, actual_output)
        r2 = r2_score(Y, actual_output)
        error_estimates = (MSE, r2)

        return actual_output, error_estimates


    def backpropagation(self):
        output_error = (self.actual_output - self.Y_data) / self.Y_data.size
        output_layer = self.layers[-1]
        breakpoint()
        error_prev = output_layer.backward_prop(output_error, self.lmbd)
        # if np.any(np.abs(error_prev) > 100):
        #     print("Wow thats really wrong")
        #     breakpoint()
        # Now backpropagate through all the hidden layers
        for l in range(self.n_hidden_layers)[::-1]:
            # if np.any(np.isinf(error_prev)):
            #     print("Gradient is fucked")
            #     breakpoint()
            hidden_layer = self.layers[l+1]
            error_hidden = hidden_layer.backward_prop(error_prev, self.lmbd)
            error_prev = error_hidden

        # Now we have all the gradients evaluated in each Layer class instance
        # Lets optimize using our GD method of choice (plain gd in this case)

        for l in range(self.n_hidden_layers):
            hidden_layer = self.layers[l+1]
            hidden_layer.weights -= self.eta * hidden_layer.grad_w
            hidden_layer.bias -= self.eta * hidden_layer.grad_b

            # if np.any(np.isnan(hidden_layer.weights)):
            #     print("weights are fucked")
            #     breakpoint()


        output_layer.weights -= self.eta * output_layer.grad_w
        output_layer.bias -= self.eta * output_layer.grad_b

    def train(self, show=False):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            print(f"Training... epoch number: {i}")
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )
                # print("training...")
                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()
            # Calculates the MSE after each epoch, so we can plot and see hwo this changes throughout the training of the NN.
            self.MSE_train.append(mean_squared_error(self.Y_data, self.actual_output))

        # Now we visualize how the MSE changes throughout the epochs
        plt.plot(np.arange(0, self.epochs), self.MSE_train)
        plt.title("MSE as a function of # epochs")
        plt.ylabel("MSE")
        plt.xlabel("Number of epochs")
        plt.legend(["MSE score"])
        if show:
            plt.show()

    def predict(self, X, Y):
        output, error_est = self.feed_forward_out(X, Y)
        return output, error_est


if __name__  == "__main__":
    # Run this to perform regression (hopefully)
    n_datapoints = 100
    f = lambda x:  8 + 2 * x + 4 * x ** 2
    x = np.linspace(0,1,n_datapoints)
    y = f(x) + 0.5 * np.random.randn(len(x))
    y = y.reshape(len(y), 1)
    deg_fit = 5
    X = np.ones((len(x), deg_fit+1))
    for i in range(1, deg_fit+1):
        X[:,i] = x ** i
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
    dnn = NeuralNetwork(X_train,Y_train,batch_size=10,eta=0.001,lmbd=0.001,epochs=50,n_hidden_neurons=25)
    dnn.train()
    pred_test, error_est_test = dnn.predict(X_test,Y_test)
    pred_train, error_est_train = dnn.predict(X_train, Y_train)
    # print("Accuracy score on training data: " + str(accuracy_score(dnn.predict(X_train[0], Y_train)[0], Y_train)))
    # print("Accuracy score on test data: " + str(accuracy_score(dnn.predict(X_test[0], Y_test)[0], Y_test)))
    breakpoint()

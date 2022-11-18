import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from Layer import Layer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn import datasets
from sys import exit

np.random.seed(0)
class NeuralNetwork():
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_layers=10,
            n_hidden_neurons=50,
            n_categories=2,
            epochs=10,
            batch_size=10,
            eta=0.1,
            lmbd=0.0,
            hidden_act_func="sigmoid",
            classifier=False):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]

        self.n_hidden_neurons = n_hidden_neurons
        self.hidden_act_func = hidden_act_func
        self.classifier = classifier
        if self.classifier:
            self.n_outputs = n_categories
        else:
            self.n_outputs = 1
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
        input_layer = Layer(self.n_features, self.n_hidden_neurons, input_layer=True)
        self.layers.append(input_layer)
        for l in range(self.n_hidden_layers):
            self.layers.append(Layer(self.n_hidden_neurons, self.n_hidden_neurons,
                                            act_func=self.hidden_act_func))

        if self.classifier:
            output_layer = Layer(self.n_hidden_neurons, self.n_outputs, output_layer=True, act_func="sigmoid")
        else:
            output_layer = Layer(self.n_hidden_neurons, self.n_outputs, output_layer=True, act_func="linear")

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
        self.output = output_layer.forward_prop(output_prev)

        if self.classifier:
            self.output[self.output<0.5] = 0
            self.output[self.output>=0.5] = 1


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
        if self.classifier:
            # actual_output[actual_output<0.5] = 0
            # actual_output[actual_output>=0.5] = 1
            error_estimate = accuracy_score(Y.reshape(actual_output.shape), actual_output)
        else:
            actual_output = output_layer.forward_prop(output_prev)
            MSE = mean_squared_error(Y, actual_output)
            r2 = r2_score(Y, actual_output)
            error_estimates = (MSE, r2)

        return actual_output, error_estimates


    def backpropagation(self):
        if self.classifier:
            breakpoint()
            output_error = (self.output - self.Y_data.reshape(self.output.shape))
        else:
            output_error = (self.output - self.Y_data.reshape(self.output.shape)) / self.Y_data.size

        output_layer = self.layers[-1]
        error_prev = output_layer.backward_prop(output_error, self.lmbd)
        weights_prev_layer = output_layer.weights
        # Now backpropagate through all the hidden layers
        for l in range(self.n_hidden_layers)[::-1]:
            hidden_layer = self.layers[l+1]
            error_hidden = hidden_layer.backward_prop(error_prev, self.lmbd, weights_prev_layer=weights_prev_layer)
            weights_prev_layer = hidden_layer.weights
            error_prev = error_hidden


        input_layer = self.layers[0]
        error_input = input_layer.backward_prop(error_prev, self.lmbd, weights_prev_layer=weights_prev_layer)

        # Now we have all the gradients evaluated in each Layer class instance
        # Lets optimize using our GD method of choice (plain gd in this case)
        input_layer = self.layers[0]
        input_layer.weights -= self.eta * input_layer.grad_w
        input_layer.bias -= self.eta * input_layer.grad_b

        for l in range(self.n_hidden_layers):
            hidden_layer = self.layers[l+1]
            hidden_layer.weights -= self.eta * hidden_layer.grad_w
            hidden_layer.bias -= self.eta * hidden_layer.grad_b

        output_layer.weights -= self.eta * output_layer.grad_w
        output_layer.bias -= self.eta * output_layer.grad_b

    def train(self, show=False):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            print(f"Training... epoch number: {i}")
            for j in range(self.iterations+1):
                # pick datapoints with replacement
                # chosen_datapoints = np.random.choice(
                #     data_indices, size=self.batch_size, replace=False
                # )

                # minibatch training data
                # self.X_data = self.X_data_full[chosen_datapoints]
                # self.Y_data = self.Y_data_full[chosen_datapoints]
                # breakpoint()
                self.X_data = self.X_data_full
                self.Y_data = self.Y_data_full
                self.feed_forward()
                self.backpropagation()
                

            if self.classifier:
                print("Accuracy score: " + str(accuracy_score(self.output, self.Y_data.reshape(self.output.shape))))
            else:
                # Calculates the MSE after each epoch, so we can plot and see hwo this changes throughout the training of the NN.
                self.MSE_train.append(mean_squared_error(self.Y_data, self.output))

        if not self.classifier:
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
    y = (y - np.min(y)) / (np.max(y) - np.min(y))           # min max scaling
    deg_fit = 2
    X = np.ones((len(x), deg_fit+1))
    for i in range(1, deg_fit+1):
        X[:,i] = x ** i
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
    dnn = NeuralNetwork(X_train,Y_train,batch_size=100,eta=0.001,lmbd=0.0,epochs=1000,n_hidden_neurons=50)
    dnn.train(show=True)
    pred_test, error_est_test = dnn.predict(X_test,Y_test)
    pred_train, error_est_train = dnn.predict(X_train, Y_train)
    # print("Accuracy score on training data: " + str(accuracy_score(dnn.predict(X_train[0], Y_train)[0], Y_train)))
    # print("Accuracy score on test data: " + str(accuracy_score(dnn.predict(X_test[0], Y_test)[0], Y_test)))
    breakpoint()

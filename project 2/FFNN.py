import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Layer import Layer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import datasets


np.random.seed(0)
class NeuralNetwork():
    """Class for Feed Forward Neural Network using SGD for optimzation

    args:
        X_data                      (np.array): Input data
        Y_data                      (np.array): Target output data
        n_hidden_layers                  (int): Number of hidden layers (default=10)
        n_hidden_neurons                 (int): Number of neurons in hidden layers (default=10)
        epochs                           (int): Number of epochs (default=10)
        batch_size                       (int): Size of minibatches (default=10)
        eta                            (float): Learning rate (default=0.1)
        lmbd                           (float): Regularization parameter (default=0.0)
        hidden_act_func               (string): Activation function for hidden layers (default=leakyrelu); options = ["leakyrelu", "sigmoid", "relu"]
        classifier                      (bool): True if network should work as a binary classifier, False if network used for regression analysis
    """
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_layers=10,
            n_hidden_neurons=10,
            epochs=10,
            batch_size=10,
            eta=0.1,
            lmbd=0.0,
            hidden_act_func="leakyrelu",
            classifier=False):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]

        self.n_hidden_neurons = n_hidden_neurons
        self.hidden_act_func = hidden_act_func
        self.classifier = classifier
        if self.classifier:
            self.n_outputs = 1
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
        self.R2score_train = []
        self.acc_score = []
        self.initialize_layers()

    def initialize_layers(self):
        """Initializes the network with all hidden layers with dimensions (n_neurons,n_neurons)
            also adds an input layer with dimensions (n_features, n_neurons)
                plus an output layer with dimensions (n_neurons, n_outputs)

            Depending on what the networks functionality is, chooses the appropriate activation function
                for the output layer.
        """
        input_layer = Layer(self.n_features, self.n_hidden_neurons, input_layer=True, act_func=self.hidden_act_func)
        self.layers.append(input_layer)
        for count, l in enumerate(range(self.n_hidden_layers)):
            self.layers.append(Layer(self.n_hidden_neurons, self.n_hidden_neurons,
                                             act_func=self.hidden_act_func))

        if self.classifier:
            output_layer = Layer(self.n_hidden_neurons, self.n_outputs, output_layer=True, act_func="sigmoid")
        else:
            output_layer = Layer(self.n_hidden_neurons, self.n_outputs, output_layer=True, act_func="linear")

        self.layers.append(output_layer)


    def feed_forward(self):
        """Feed Forward stage for the network. This call the feed_forward function in each layer
                that takes the previous input and produces the output going into the next layer.
        """
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


    def feed_forward_out(self, X, Y):
        """Feed forward stage for predicition of the network. Takes also input data
                X_test and Y_test (if traintestsplitting your data)

        args:
            X                      (np.array): Array of X-values (input data to the network)
            Y                      (np.array): True target that we want the network to predict (used to evaluate error scores)

        returns:
            output                 (np.array): Array of shape = Y.shape that has networks prediction
            error_estimates    (tuple)/(dict): Tuple containing MSE and R2 score/if classifier = dict contaning test_data and train_data accuracy scores

        """
        input_layer = self.layers[0]
        output_prev = input_layer.forward_prop(X)
        # Feed through hidden layers
        for l in range(self.n_hidden_layers):
            hidden_layer = self.layers[l+1]
            output = hidden_layer.forward_prop(output_prev)
            output_prev = output

        # And through output layer
        output_layer = self.layers[-1]
        output = output_layer.forward_prop(output_prev)
        # Now finding error estimates for either classifier or regression
        if self.classifier:
            # Hard classifier
            output[output<0.5] = 0
            output[output>=0.5] = 1
            self.output[self.output<0.5] = 0
            self.output[self.output>=0.5] = 1

            error_est_test = accuracy_score(Y.reshape(output.shape), output)
            error_est_train = accuracy_score(self.Y_data.reshape(self.output.shape), self.output)
            error_estimates = {"test":error_est_test, "train": error_est_train}
        else:
            MSE = mean_squared_error(Y, output)
            r2 = r2_score(Y, output)
            error_estimates = (MSE, r2)

        return output, error_estimates


    def backpropagation(self):
        """Performs the optimization of the network using backpropagation algorithm.
            The gradient is optimized using Ridge (if lmbda = 0 regular OLS), and SGD is chosen for the
                actual optimization. The batching of data happens in sel.train() method.

            The cost function is different for the classifier and regression case.

        """
        if self.classifier:
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
        # Lets optimize using our GD method of choice (SGD in this case)
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
        """Trains the network by first minibatching the data and then calling the
            feed_forward and backpropagation methods in each iteration.
                Has functionality to visualize the MSE of the network for every epoch
                    to show how the network is training properly.
        """
        data_indices = np.arange(self.n_inputs)         # Finds the indexes available from our dataset

        for i in range(self.epochs):
            for j in range(self.iterations+1):
                # pick datapoints with replacement randomly
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()


            if self.classifier:
                pass
            else:
                # Calculates the MSE after each epoch, so we can plot and see hwo this changes throughout the training of the NN.
                self.MSE_train.append(mean_squared_error(self.Y_data.reshape(self.output.shape), self.output))
                self.R2score_train.append(r2_score(self.Y_data.reshape(self.output.shape), self.output))

        if not self.classifier and show:
            # Now we visualize how the MSE and R2 score changes throughout the epochs
            # This code will produce plot shown in result section (Training network)
            fig, axs = plt.subplots(nrows=1,ncols=2, figsize=(8,6))
            fig.suptitle("Training the network: Backpropagation")
            axs[0].plot(np.arange(0, len(self.MSE_train)), self.MSE_train, color="darkcyan")
            axs[0].set_title("MSE")
            axs[0].set_ylabel("MSE")
            axs[0].set_xlabel("Number of epochs")
            axs[0].legend(["MSE score"])

            axs[1].plot(np.arange(0, len(self.R2score_train)), self.R2score_train, color="darkcyan")
            axs[1].set_title("R2 score")
            axs[1].set_ylabel("R2 score")
            axs[1].set_xlabel("Number of epochs")
            axs[1].legend(["R2 score"])
            fig.tight_layout()
            # plt.savefig("e100_eta001_lmbd000001_nneu5_nhl5_train_network_mser2.pdf")
            plt.show()

    def predict(self, X, Y):
        output, error_est = self.feed_forward_out(X, Y)
        return output, error_est


if __name__  == "__main__":
    np.random.seed(0)
    # Run this to perform regression (hopefully)
    sns.set_theme()
    n_datapoints = 1000
    f = lambda x:  2 * x + 4 * x ** 2 - 2 * x ** 3
    x = np.linspace(-3,3,n_datapoints).reshape(n_datapoints,1)
    y = f(x)
    scaler = MinMaxScaler()                     # Min max scaling
    x, y = scaler.fit_transform(x), scaler.fit_transform(y) + (0.02 * np.random.randn(len(x))).reshape(f(x).shape)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
    eta_vals = np.logspace(-6, 0, 7)
    lmbd_vals = np.logspace(-6, 0, 7)
    if False: # This code finds the optimal parameters lambda and eta  by using sns.heatmap
        eta_vals = np.logspace(-6, 0, 7)
        lmbd_vals = np.logspace(-6, 0, 7)
        mse = np.zeros((7,7))
        r2 = np.zeros((7,7))

        for i, eta in enumerate(eta_vals):
            for j, lmbd in enumerate(lmbd_vals):
                dnn = NeuralNetwork(X_train,Y_train,batch_size=100,eta=eta,
                            lmbd=lmbd,epochs=10000,n_hidden_neurons= 4, n_hidden_layers=1)
                dnn.train()
                mse[i][j] = dnn.MSE_train[-1]
                r2[i][j] = dnn.R2score_train[-1]
        sns.heatmap(mse,annot=True)
        sns.heatmap(r2,annot=True)

        optimal = (eta_vals[5], lmbd_vals[0])

    if False: # Finds the R2 and MSE score for the FFNN regression model by different methods (change it in class name) as presetned in result sect. 3.2.1
        dnn = NeuralNetwork(X_train,Y_train,batch_size=100,eta=0.1,
                    lmbd=0.000001,epochs=10000,n_hidden_neurons= 4, n_hidden_layers=1)
        dnn.train(show=False)
        # true_curve = scaler.fit_transform(f(X_test)) + (0.02 * np.random.randn(len(X_test))).reshape(f(X_test).shape)
        output, er_est = dnn.predict(X_test, Y_test)
        reg = LinearRegression().fit(X_train,Y_train)
        linreg_score = reg.score(X_test, Y_test)
        ylinreg = reg.predict(x)
        fig, ax = plt.subplots()
        ax.plot(x, ylinreg, "darkcyan", x[::20],y[::20], "ro")
        ax.set_title("Scikit-Learn LinReg data vs true function data")
        ax.set_ylabel("y - axis", fontsize=10)
        ax.set_xlabel("x - axis", fontsize=10)
        ax.legend(["LinReg model", "True model"])
        # plt.savefig("linreg_vs_truedata.pdf")
        breakpoint()

        # fig, ax = plt.subplots()
        # ax.plot(x, output,"darkcyan", x[::20], y[::20], "ro",)
        # ax.set_xlabel("x - values", fontsize=10)
        # ax.set_ylabel("y - values", fontsize=10)
        # ax.set_title("FFNN model function data vs. true function data")
        # ax.legend(["FFNN model", "True model"])
        # plt.savefig("output_FFNN_vs_truedata.pdf")
        # plt.show()        # This code will produce the data in the table in section (Comparison between act func)


    if False:  # Will produce the graph fit to the data, presented in result section (3.2)
        dnn = NeuralNetwork(X_train,Y_train,batch_size=100,eta=0.1,
                    lmbd=0.000001,epochs=500,n_hidden_neurons= 5, n_hidden_layers=5)
        dnn.train(show=False)
        output, er_est = dnn.predict(X_test, Y_test)
        poutput, asdf = dnn.predict(x,y)
        # plt.plot(x,y,"darkcyan",x,poutput,"k-")
        # plt.title("Visualization of the FFNN model fit onto our function data")
        # plt.xlabel("x - axis", fontsize=10)
        # plt.ylabel("y - axis", fontsize=10)
        # plt.legend(["True data", "FFNN model"])
        # plt.savefig("20neuron_5hidden.pdf")
        # plt.show()
        print(er_est)

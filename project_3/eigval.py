import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

np.random.seed(20)
def exact_solution(A):
    """Calculates the eigenvalues of a square-matrix "A"
        by np.linalg.eigvals functionality.

    args:
        A           (np.array): Square matrix, i.e NxN

    returns:
        eigvals     (np.array): Eigenvalues of the matrix A
    """

    eigvals, eigvecs = np.linalg.eig(A)
    return eigvals, eigvecs

class TFNeuralNetwork():
    """Class to solve for highest (or lowest) eigenvalue
        of a NxN symmetric, square matrix A using tensorflows Neural Network.

        The loss function is MSE(x(t), f(x(t))) as presented in Yi et al, and we use
            Adam optimizer to minimize this cost function.

    args:
        matrix                  (np.array): The NxN symmetric matrix we want to compute eigenvalue of
        n_hidden_layers         (int)     : Number of hidden layers in the Neural Network
        n_hidden_neurons        (int)     : Number of hidden neurons in the hidden layers
        max_iterations          (int)     : Maximum number of iterations before stopping the networks search for eigenvalue
        learning_rate           (float)   : Learning rate used in the ADAM optimizer
        act_func                (str)     : Activation function to be used in the hidden layers
    """
    def __init__(self,
            matrix,
            n_hidden_layers = 10,
            n_hidden_neurons = 15,
            max_iterations = 15,
            learning_rate = 0.0001,
            act_func = "relu",
            ):

        self.matrix_shape = matrix.shape
        self.matrix = tf.convert_to_tensor(matrix, dtype="float32")
        self.n_iterations = max_iterations
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.activation_function = act_func
        self.lr = learning_rate
        x0 = np.random.randn(len(matrix))                  # Initial guess of eigenvector (will always converge regardless of initial guess)
        self.x0 = tf.convert_to_tensor(x0)                 # As explained in Yi et al.
        self.x0 = tf.reshape(self.x0, (6,1))


        self.initialize_network()
        self.solving()

    @tf.function
    def f(self, x, A):
        """When we have converged to solution, f = x(t), i.e dxdt = 0
                as explained in Yi et al section 1.
        args:
            x       (tf.tensor): Output of the network
            A        (np.array): Matrix to solve for eigenvalues

        returns:
            f       (tf.tensor): Output of the function f(x(t)), as presented in the paper by Yi et al
        """

        first_term = tf.tensordot(tf.transpose(x), x, axes=1) * A
        second_term = (1 - tf.tensordot(tf.transpose(x), tf.tensordot(A, x, axes=1),axes=1) ) * np.eye(*self.matrix_shape)
        f = tf.tensordot((first_term + second_term), x, axes=1)

        return f

    def initialize_network(self):
        """Sets up the Neural Network architecture
            with the input-, hidden- and output layer(s)

        Implied that the Neural Network has atleast 1 hidden layer, and the first addition of a hidden layer
            also creates the input layer.
        I.e the architecture will be:
            - Input layer
            - # of hidden layers, where # = n_hidden_layers with activation function = act_func (default is ReLu)
            - Output layer, with linear activation function.
        """
        self.model = Sequential()
        for i in range(self.n_hidden_layers):                                   # Adds the hidden layers, where the first layer (i==0), also add the input layer
            if i == 0:
                self.model.add(Dense(self.n_hidden_neurons, input_shape=(len(self.matrix),1), activation=self.activation_function))
            else:
                self.model.add(Dense(self.n_hidden_neurons, activation=self.activation_function))
        self.model.add(Dense(1, activation="linear"))                           # The final hidden layer, output layer

    @tf.function
    def loss_fn(self, y_true, y_pred):
        """Calculates the Mean squared error
            of two tf.tensorobjects

        args:
            y_true      (tf.tensor): Target values
            y_pred      (tf.tensor): Model values

        returns:
            MSE         (tf.tensor): MSE value (scalar value)
        """
        loss = tf.reduce_mean(tf.square(y_pred - y_true))

        return loss


    def solving(self):
        """Trains the network, while searching for the eigenvalue of the matrix.

        Keeps track of the loss function in each iteration, to produce a plot of convergence.
        Also keeps track of the eigenvalue, which is computed from the eigenvector output of the network
            using Reyleigh quotient.

        """
        self.loss = []
        self.eigvals = []
        tol = 1e-16
        x_trial = tf.reshape(self.model(self.x0), (6,1))
        self.loss.append(self.loss_fn(x_trial, self.f(x_trial, self.matrix)))        # Adds a duplicate of the initial loss, to start the loop.
        optimizer = optimizers.Adam(learning_rate=self.lr)
        iterations = 0                                                               # Counts number of iterations (called epochs, which is slightly misleading)
        while self.loss[-1] > tol and iterations < self.n_iterations:
            with tf.GradientTape() as tape:
                if iterations == 0:
                    x_trial = tf.reshape(self.model(self.x0), (6,1))
                else:
                    x_trial = tf.reshape(self.model(x_trial), (6,1))

                self.f_trial = self.f(x_trial, self.matrix)
                self.eigval_trial = (tf.transpose(x_trial) @ (self.matrix @ x_trial)) / (tf.transpose(x_trial) @ x_trial)
                self.eigvec_trial = x_trial
                loss = self.loss_fn(x_trial, self.f_trial) #+ self.loss_fn(self.eigval_trial, 30)

            self.loss.append(float(loss))
            self.eigvals.append(float(self.eigval_trial.numpy()))

            self.gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(
            self.gradients, self.model.trainable_variables
            ))

            # if iterations % 100 == 0:
            print(f"Iteration number {iterations}. Current eigenvalue estimate: {float(self.eigval_trial)}")
            iterations += 1
        self.loss.pop(0)                          # Removes the duplicate of the first loss, added prior to the loop
        self.eigenvalue = self.eigval_trial
        self.eigenvector = self.eigvec_trial

if __name__ == "__main__":
    sns.set_theme()
    ## The square, real, symmetric, 6x6 matrix we use for our calculations - to calculate lowest eigenvalue, swap A for -A inside the network. Theorem 4-5 in Yi et al
    matrix = np.array([
    [5, 2, 3, 2, 6, 9],
    [2, 9, 8, 1, 2, 1],
    [3, 8, 7, 5, 0, 4],
    [2, 1, 5, 1, 8, 3],
    [6, 2, 0, 8, 7, 0],
    [9, 1, 4, 3, 0, 9]]
    )

    eigvals, eigvecs = exact_solution(matrix)
    print("Eigenvalues are: ", eigvals, "\n")
    dnn = TFNeuralNetwork(matrix=matrix)
    eigval = dnn.eigval_trial
    eigvec = dnn.eigvec_trial
    fig, axs = plt.subplots(2, figsize=(8,8))
    axs[0].plot(np.arange(0, len(dnn.loss), 1), dnn.loss)
    axs[0].set_title("Loss as function of iteration")
    axs[0].set_ylabel("Loss estimate", fontsize=10)
    axs[0].set_xlabel("Number of iterations", fontsize=10)
    axs[1].plot(np.arange(0, len(dnn.eigvals), 1), dnn.eigvals, np.arange(0, len(dnn.eigvals), 1), eigvals[0]*np.ones(len(dnn.eigvals)), "r--")
    axs[1].set_title("Convergence rate to eigenvalue estimate")
    axs[1].set_ylabel("Eigenvalue", fontsize=10)
    axs[1].set_xlabel("Number of iterations", fontsize=10)
    fig.tight_layout()
    # plt.savefig("eigval_NN.pdf")
    plt.show()

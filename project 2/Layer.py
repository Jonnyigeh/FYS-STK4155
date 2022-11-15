import numpy as np


class Layer():
    """ Classobject that will construct a single layer of a Neural Network. This will save
    various parameters such as gradients, biases, weights etc. that belong in each layer.

    args:
        dimension_in            (int): Left side dimension of the weights matrix
        dimension_out           (int): Right side dimension of the weights matrix and the width of the bias vector
        act_func                (string): Specifies which activation function to be used in the layer (sigmoid, relu or linear)
        input_layer             (bool): True if the layer is an input layer
        output_layer            (bool): True if the layer is an output layer

    """
    def __init__(
            self,
            dimension_in,
            dimension_out,
            act_func="relu",
            input_layer=False,
            output_layer=False,
            ):

        self.dimension_in = dimension_in
        self.dimension_out = dimension_out
        self.act_func = act_func
        self.output_layer = output_layer

        if input_layer:             # Input layer does not change the input, just changes the dimensionality.
            self.weights = np.eye(self.dimension_in, self.dimension_out)
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
            return np.where(x>0, 1, 0.01 * x) # Leaky ReLu

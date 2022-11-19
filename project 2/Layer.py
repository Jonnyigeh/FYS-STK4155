import numpy as np


class Layer():
    """ Classobject that will construct a single layer of a Neural Network. This will save
    various parameters such as gradients, biases, weights etc. that belong in each layer.

    args:
        dimension_in            (int):    Left side dimension of the weights matrix
        dimension_out           (int):    Right side dimension of the weights matrix and the width of the bias vector
        act_func                (string): Specifies which activation function to be used in the layer (sigmoid, relu or linear)
        input_layer             (bool):   True if the layer is an input layer
        output_layer            (bool):   True if the layer is an output layer

    """
    def __init__(
            self,
            dimension_in,
            dimension_out,
            act_func="sigmoid",
            input_layer=False,
            output_layer=False,
            ):

        self.dimension_in = dimension_in
        self.dimension_out = dimension_out
        self.act_func = act_func
        self.output_layer = output_layer
        self.input_layer = input_layer

        heuristic = np.sqrt(2 / self.dimension_out)     # He-et-al initilization
        self.weights = np.random.randn(self.dimension_in, self.dimension_out)  * heuristic
        self.bias = np.zeros((1, self.dimension_out)) + 0.01

    def forward_prop(self, input_value):
        """Forward feed through the layer.

        args:
            input_value     (np.array): Input data to the layer

        returns:
            self.output     (np.array): Output from the layer
        """
        self.input = input_value
        self.val = self.input @ self.weights + self.bias
        self.output = self.activation_function(self.val)

        return self.output

    def backward_prop(self, error_prev_layer, lmbda, weights_prev_layer = 0):
        """Backpropagation through the layer

        args:
            error_prev_layer         (np.array):
            lmbda                       (float):
            weights_prev_layer       (np.array):

        returns:
            error_prev_layer         (np.array): If layer is an output layer, return the error between output and target (calculated outside the layer)
            self.error_layer         (np.arrya): Error in the layer
        """
        if self.output_layer:
            self.grad_w = self.input.T @ error_prev_layer
            self.grad_b = np.sum(error_prev_layer, axis=0, keepdims=True)
            if lmbda > 0.0:
                self.grad_w += lmbda * self.weights

            return error_prev_layer

        else:
            self.error_layer = (error_prev_layer @ weights_prev_layer.T) * self.grad_activation_function(self.output)
            self.grad_w = self.input.T @ self.error_layer
            self.grad_b = np.sum(self.error_layer, axis=0, keepdims=True)
        if lmbda > 0.0:
            self.grad_w += lmbda * self.weights

        return self.error_layer

    def activation_function(self, x):
        """Activation function on the layer

        args:
            x           (np.array): input data to the activation function

        returns:
                        (np.array): Output from the given activation function (dependant on which function is true)

        """
        if self.act_func=="sigmoid":
            return 1 / (1 + np.exp(-x))

        if self.act_func=="linear":
            return x

        if self.act_func=="relu":
            return np.maximum(0,x)

        if self.act_func=="leakyrelu":
            return np.where(x<0, 0.01 * x, x)

    def grad_activation_function(self, x):
        """Gradient of the activation function wrt. input

        args:
            x           (np.array): input data to the activation function

        returns:
                        (np.array): Output from the gradient of activation function (dependant on which function is true)
        """
        if self.act_func=="sigmoid":
            return self.activation_function(x) * ( 1 - self.activation_function(x))

        if self.act_func=="linear":
            return np.ones_like(x)

        if self.act_func=="relu":
            return np.where(x>0, 1, 0)

        if self.act_func=="leakyrelu":
            return np.where(x>0, 1, 0.01)

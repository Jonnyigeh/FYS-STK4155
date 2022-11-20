## Readme file for instruction to run various algorithms in project 2: FYS-STK4155


### Rate_of_conv_SGD.py
- Run the code (a snippet from _part A kode_) to simply produce the graphs presented in result section.

### FFNN.py
Class module for the feed-forward neural network. 
- In the if name block, if False statements can be changed to if True to produce the various data we've presented in the result section. Which statements produce which results have been written as comments. 

### classifier_FFNN.py
- This imports the FFNN class module NeuralNetwork and the if name block produces the heatmap, and tables (again if False -> if True), presented in the result section.

### Layer.py
Class module for the layers in the NeuralNetwork class. Holds attributes for weights, biases, output etc. Does not need to be run, will be imported into the relevant scripts (classifier etc.)

### Gradient_methods.py

Class module for gradient descent and all its varying methods inside the class module. 
- Run the code in if name block and set each if statement to True if you want to run for each result, which is also commented in the code. 
- Set given if statement to True RIDGE values to run either if statement for MSE of each method or heatmap plot.


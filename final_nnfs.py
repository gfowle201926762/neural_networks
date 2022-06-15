# Neural network from scratch, using numpy properly.

import numpy as np
import pandas as pd
from pydataset import data
from matplotlib import pyplot as plt

import spiral_data

"""df = data('iris')

input_names = df.columns[:4].to_numpy()
inputs = df.iloc[:, :4]
inputs = inputs.to_numpy()

y = df.iloc[:, 4]
y = y.to_numpy()

label_names = np.unique(y)

y = np.where(y=='setosa', 0, y)
y = np.where(y=='versicolor', 1, y)
y = np.where(y=='virginica', 2, y)
y = y.astype(int)"""

#inputs = np.array([[1, 2, 3, 1], [3, 2, 1, 1], [0, 0, 1, 1], [5, 1, 1, 1]])                     # preparing for a batch of inputs
#dvalues = np.array([[1, 2, 3], [1, 1, 1], [2, 2, 2], [1, 2, 1], [3, 2, 3]])        # batch of derivative base values. They act as a substitute for a loss function. I'm guessing their lengths must be the same as the inputs.
#y = np.array([1, 2, 0, 1])                                     # which class is correct for each sample... But, they could also be in one-hot-encoding







class DataPreparation: # this should only be used for labeled data.
    def __init__(self, df):
        # Initialise the object using a dataframe which contains all the data we want to use.
        # Seperate the inputs, labels, input names, and class names into their own numpy arrays.
        self.no_columns = len(df.columns)
        self.no_samples = len(df)
        self.no_inputs = self.no_columns - 1

        self.labels = df.iloc[:, self.no_columns - 1].to_numpy()
        self.label_names = np.unique(self.labels)
        self.no_classes = len(self.label_names)
        replacement_labels = np.arange(0, self.no_classes)
        for i in range(self.no_classes):
            self.labels = np.where(self.labels==self.label_names[i], replacement_labels[i], self.labels)

        self.labels = self.labels.astype(int)

        self.input_names = df.columns[:self.no_inputs].to_numpy()
        self.inputs = df.iloc[:, :self.no_inputs].to_numpy()

    def randomise(self):
        self.samples = np.column_stack((self.inputs, self.labels))
        np.random.shuffle(self.samples)


    def split(self, testing_pc):
        # return two objects, train and test. Both of these objects have the attributes .x and .y.
        self.randomise()

        testing_index = round(self.no_samples * testing_pc)
        self.test_data = self.samples[:testing_index].astype(int)
        self.training_data = self.samples[testing_index:].astype(int)

        self.test_data_x = self.test_data[:, :self.no_inputs].astype(int)
        self.test_data_y = self.test_data[:, self.no_columns - 1].astype(int)

        self.train_data_x = self.training_data[:, :self.no_inputs].astype(int)
        self.train_data_y = self.training_data[:, self.no_columns - 1].astype(int)









class Layer_Dense:
    def __init__(self, previous, current):
        self.previous = previous
        self.current = current
        self.weights = np.random.normal(loc=0, scale=0.1, size=(previous, current))    # SHAPE (neurons in previous layer, neurons in current layer)
        self.biases = np.random.normal(loc=0, scale=0.1, size=(current))               # SHAPE (neurons in current layer)

    def forward(self, inputs):
        self.inputs = inputs                                            # remember the inputs values for the backward pass
        self.output = np.dot(self.inputs, self.weights)                 # returns an array. Each element is a list corresponsing to the output for a single set of inputs.

    def backward(self, dvalues):

        ### BACKWARD PASS:
        # Each neuron in the current layer must receive a base case (dvalues). 
        # This is the derivative of each neuron's output with respect to loss.
        # Therefore, this layer receives a matrix which has the same rows as the number of neurons in the current layer, and columns as number of samples.

        ### DERIVING INPUTS OF CURRENT LAYER:
        # This layer wants to pass on the derivative of the loss with respect to its own inputs to the previous layer.
        # These derivatives are summed to make a vector of the same length as the previous layer of neurons. In a batch, multiple vectors become an array.
        # It wants to be in the same (kind of) shape this layer received its dvalues in.
        # Dvalues has the same number of neurons in the current layer, and weights are corresponding to the number of neurons in the current layer.
        # So, we need to transpose weights so their shapes align for the np.dot()
        
        #c = np.dot(self.weights, dvalues.T)                            # This also produces an answer because the shapes align correctly. However, we do not want this, because the lowest dimension wants to be how many neurons there are in the previous layer.
        self.dinputs = np.dot(dvalues, self.weights.T)                  # This is correct.
        

        ### DERIVING WEIGHTS OF CURRENT LAYER:
        # This layer wants the derivative of the loss with respect to its own weights.
        # These must be in the same shape as self.weights: previous neurons as columns, current neurons as rows.

        #a = np.dot(dvalues.T, self.inputs)                             # This has correctly aligned dimensions, but the shape is not the same as self.weights
        self.dweights = np.dot(self.inputs.T, dvalues)                  # This is correct.


        ### DERIVING BIASES OF CURRENT LAYER:
        # It must be in the same shape as self.biases
        # da_db is always 1, so we don't need to worry about it.
        # If it is a single sample, there is no summing. If it is a batch, each dvalue which corresponds to the same neuron needs to be summed together.
        
        self.dbiases = np.sum(dvalues, axis=0)                          # This is correct.



class Activation_Relu:
    def forward(self, inputs):
        self.inputs = inputs                                            # remember the inputs for the backward pass.
        self.output = np.maximum(inputs, 0)

    def backward(self, dvalues):
        # This will receive the derivative of the loss with respect to the activation's output, as dvalues.
        # This should be in the same shape as the number of neurons in the current layer, and also the same shape as self.inputs.
        # It should pass on something with the same shape as dvalues (just going to the actual Layer.)
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0                              # This indexes the dinputs array at the same relative location as the inputs array, changing dinputs to 0 if inputs <= 0.



class Activation_Linear:
    def forward(self, inputs):
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues



class Activation_Softmax:
    def forward(self, inputs):
        # receiving a batch of inputs the length of a layer
        # it should forward outputs which are the same shape as its inputs.
        self.inputs = inputs                                            # save the inputs for backpropagation (probably)

        new = np.exp(np.subtract(inputs, np.max(inputs, axis=1, keepdims=True))) # Subtracting the maximum value from all the inputs in each sample to make the max values 0. Then exponentiate it.
        summed = np.sum(new, axis=1, keepdims=True)                     # Sum the outputs of each sample together (to prepare for normalisation)
        self.output = np.divide(new, summed)                            # normalise



class Loss_CategoricalCrossEntropy:                                     # This should be used to calculate loss for classification problems, usually with a softmax output.
    def calculate_loss(self, predicted, y):
        # We receive softmax output in predicted, and the ground truth values in y.
        # We only need the softmax output where its index is equal to the index where y = 1 in order to calculate loss.

        if len(y.shape) == 1: # not one-hot
            confidence = predicted[range(len(predicted)), y]            # iterating through every column in predicted (through every sample), and then taking the index of y - 1 (assuming y starts its index at 1).

        elif len(y.shape) == 2: # one-hot encoded
            confidence = predicted[range(len(predicted)), np.argmax(y, axis=1)] # iterates through as before, and takes the index of where the highest value is in y (the 1).

        clipped = np.clip(confidence, 1e-10, 1 - 1e-10)                 # clip the values to prevent calculating ln(0)
        self.loss = -np.log(clipped)
        return np.mean(self.loss)



class Accuracy:
    def calculate_accuracy(self, predicted, y):
        # calculate whether it is predicting correctly or not.
        
        # convert into one-hot
        if len(y.shape) == 1:
            new = np.zeros((y.size, prepared_data.no_classes))
            new[np.arange(y.size), y] = 1
            y = new
        
        # Convert into vectors where the elements are the indices of what was predicted and what is the ground truth.
        predicted = np.argmax(predicted, axis=1)
        y = np.argmax(y, axis=1)

        correct = np.sum(y == predicted)
        self.accuracy = correct / len(y)

        return self.accuracy




class Back_SoftmaxCategoricalCrossEntropy:
    def backward(self, predicted, y, classes):
        # We receive softmax output in predicted, and the ground truth values in y.
        # we need to pass on a matrix which has the same shape as the output layer (corresponding number of neurons).
        # We DO NOT need the value of loss to calculate this backward pass - see the combined derivatives of softmax and cce.

        # Turn into one hot encoding if not. This might be a bit inefficient, because then we are subtracting 0 from a lot of values. oh well.
        if len(y.shape) == 1:
            new = np.zeros((y.size, classes))
            new[np.arange(y.size), y] = 1
            y = new

        self.dinputs = predicted - y
        self.dinputs = self.dinputs / len(predicted)                    # Normalise the dinputs, to make the sum of these gradients invariant to the number of samples we pass through



class Optimiser_StochasticGradientDescent:
    # we want this to update the weights and biases of the model.
    def __init__(self, learning_rate, decay_rate):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

    def update(self, layer, epoch):
        if epoch == 0:
            epoch = 1
        layer.weights -= (layer.dweights * (self.learning_rate / (self.decay_rate * epoch)))
        layer.biases -= (layer.dbiases * (self.learning_rate / (self.decay_rate * epoch)))



class ModelCreator:
    def __init__(self, no_inputs, no_outputs, no_depth, no_height, hidden_activation, output_activation, loss, optimiser, learning_rate, decay_rate):
        self.no_depth = no_depth
        self.no_outputs = no_outputs
        self.no_inputs = no_inputs
        self.no_height = no_height
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimiser = optimiser
        
        self.layers = []
        for layer in range(no_depth):
            if layer == 0 and no_depth > 1:
                self.layers.append(Layer_Dense(no_inputs, no_height))
            elif layer == 0 and no_depth == 1:
                self.layers.append(Layer_Dense(no_inputs, no_outputs))
            elif layer == no_depth - 1:
                self.layers.append(Layer_Dense(no_height, no_outputs))
            else:
                self.layers.append(Layer_Dense(no_height, no_height))

        self.activations = []
        for activation in range(no_depth):
            if activation == no_depth - 1:
                if output_activation == "softmax":
                    self.activations.append(Activation_Softmax())
            else:
                if hidden_activation == "relu":
                    self.activations.append(Activation_Relu())
                elif hidden_activation == "linear":
                    self.activations.append(Activation_Linear())

        if loss == "cce":
            self.loss_object = Loss_CategoricalCrossEntropy()
            self.back_loss_object = Back_SoftmaxCategoricalCrossEntropy()

        if optimiser == "sgd":
            self.optimiser_object = Optimiser_StochasticGradientDescent(learning_rate, decay_rate)

        self.accuracy_object = Accuracy()

    def generate_vis(self):
        self.vis_layers = [self.no_inputs]
        for layer in self.layers:
            self.vis_layers.append(layer.current)

        self.max_height = max(self.no_height, self.no_inputs, self.no_outputs)

    def train(self, data, labels, epoch):
        for i in range(self.no_depth):
            self.layers[i].forward(data)
            self.activations[i].forward(self.layers[i].output)
            data = self.activations[i].output
        
        self.loss = self.loss_object.calculate_loss(data, labels)
        self.acc = self.accuracy_object.calculate_accuracy(data, labels)
        self.back_loss_object.backward(data, labels, self.no_outputs)

        for i in reversed(range(self.no_depth)):
            if i == self.no_depth - 1:
                self.layers[i].backward(self.back_loss_object.dinputs)
                data = self.layers[i].dinputs
            else:
                self.activations[i].backward(data)
                self.layers[i].backward(self.activations[i].dinputs)
                data = data = self.layers[i].dinputs

        for i in range(self.no_depth):
            self.optimiser_object.update(self.layers[i], epoch)

    def go(self, epochs, data, labels):
        x = 0
        while x <= epochs:
            self.train(data, labels, x)
            if x % 100 == 0:
                print(  f"Epoch: {x}   " +
                        f"Loss: {self.loss:.4f}   " +
                        f"Accuracy: {self.acc:.4f}")
            x += 1

    def vis_go(self, epochs, data, labels):
        self.train(data, labels)
        if epochs % 100 == 0:
            print(  f"Epoch: {epochs}   " +
                    f"Loss: {self.loss:.4f}   " +
                    f"Accuracy: {self.acc:.4f}")
        return self.loss, self.acc, epochs

    def test(self, data, labels):
        for i in range(self.no_depth):
            self.layers[i].forward(data)
            self.activations[i].forward(self.layers[i].output)
            data = self.activations[i].output
        
        self.loss = self.loss_object.calculate_loss(data, labels)
        self.acc = self.accuracy_object.calculate_accuracy(data, labels)
        self.back_loss_object.backward(data, labels, self.no_outputs)

        print(  "\nTESTING:\n" +
                f"Loss: {self.loss:.4f}   " +
                f"Accuracy: {self.acc:.4f}\n")

    def map(self, data):
        mapping = []
        for x in data:
            z = x.copy()
            for i in range(self.no_depth):
                self.layers[i].forward(z)
                self.activations[i].forward(self.layers[i].output)
                z = self.activations[i].output
            a = []
            for num in range(self.no_inputs):
                a.append(x[0, num])

            a.append(np.argmax(z))
            mapping.append(a)

        return np.array(mapping)



datacreator = spiral_data.SpiralDataCreator(1000, type="strange")
df = datacreator.df
prepared_data = DataPreparation(df)
prepared_data.split(testing_pc=0.1)



train_x = prepared_data.train_data_x
train_y = prepared_data.train_data_y

test_x = prepared_data.test_data_x
test_y = prepared_data.test_data_y

no_classes = prepared_data.no_classes
no_inputs = prepared_data.no_inputs

model = ModelCreator(no_inputs, no_classes, 3, 16, "relu", "softmax", "cce", "sgd", learning_rate=1, decay_rate=1)

print(f"INPUTS:{model.no_inputs}")
print(f"CLASSES:{model.no_outputs}")

model.go(1000, train_x, train_y)
model.test(test_x, test_y)




test = np.array([[[-60, -60]]])
for a in range(-60, 61, 2):
    for b in range(-59, 61, 2):
        add = np.array([[[a, b]]])
        test = np.concatenate((test, add), axis=0)

mapping = model.map(test)

set0_index = np.where(mapping[:, 2] == 0)[0]
set0 = mapping[set0_index]

set1_index = np.where(mapping[:, 2] == 1)[0]
set1 = mapping[set1_index]

set2_index = np.where(mapping[:, 2] == 2)[0]
set2 = mapping[set2_index]

x_0 = set0[:, 0]
y_0 = set0[:, 1]

x_1 = set1[:, 0]
y_1 = set1[:, 1]

x_2 = set2[:, 0]
y_2 = set2[:, 1]

plot_size = 4

map_0 = plt.scatter(x_0, y_0, s=plot_size)
map_1 = plt.scatter(x_1, y_1, s=plot_size)
map_2 = plt.scatter(x_2, y_2, s=plot_size)

x_a = datacreator.x_a
x_b = datacreator.x_b
x_c = datacreator.x_c
data_0 = plt.scatter(x_a[:,0],x_a[:,1], s=plot_size)
data_1 = plt.scatter(x_b[:,0],x_b[:,1], s=plot_size)
data_2 = plt.scatter(x_c[:,0],x_c[:,1], s=plot_size)

plt.legend((map_0, data_0, map_1, data_1, map_2, data_2),
           ('map_0', 'data_0', 'map_1', 'data_1', 'map_2', 'data_2'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)

plt.show()










'''layer_1 = Layer_Dense(no_inputs, 6)
layer_2 = Layer_Dense(6, classes)

all_layers = [layer_1, layer_2]
layers = [len(inputs[0]), all_layers[0].current, all_layers[1].current]

relu_1 = Activation_Relu()
relu_2 = Activation_Relu()
relu_3 = Activation_Relu()
relu_4 = Activation_Relu()

softmax = Activation_Softmax()
categorical_cross_entropy = Loss_CategoricalCrossEntropy()
backsoftmaxcce = Back_SoftmaxCategoricalCrossEntropy()

optimiser_SGD = Optimiser_StochasticGradientDescent(learning_rate=0.1)

accuracy = Accuracy()

def go(x):

    ### FORWARD PASS ###
    layer_1.forward(inputs)
    relu_1.forward(layer_1.output)

    layer_2.forward(relu_1.output)
    softmax.forward(layer_2.output)

    ### LOSS CALCULATION ###
    loss = categorical_cross_entropy.calculate_loss(softmax.output, y)

    ### ACCURACY CALCULATION ###
    acc = accuracy.calculate_accuracy(softmax.output, y)

    ### BACKWARD PASS ###
    backsoftmaxcce.backward(softmax.output, y, classes)
    layer_2.backward(backsoftmaxcce.dinputs)

    relu_1.backward(layer_2.dinputs)
    layer_1.backward(relu_1.dinputs)

    ### OPTIMISATION ###
    optimiser_SGD.update(layer_1)
    optimiser_SGD.update(layer_2)


    if x % 10 == 0:
        print(  f"Epoch: {x}   " +
                f"Loss: {loss:.4f}   " +
                f"Accuracy: {acc:.4f}")

    return loss, acc, x


"""x = 0
while x <= 1000:
    go(x)
    x += 1"""'''
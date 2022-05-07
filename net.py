import numpy
import random

# create a class which makes Layers.
# inputs: number of nodes in the previous layer, number of nodes in the current layer. Don't worry about batches yet.

inputs = numpy.array([[14, 21, 32], [0.1, 0.3, 0.2], [5, 6, 4]])

labels = numpy.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) # ONE HOT ENCODING

class Layer:
    def __init__(self, inputs, size):
        self.inputs = inputs # number of nodes in the previous layer
        self.size = size # number of nodes in the current layer
        self.biases = numpy.random.normal(loc=0, scale=1, size=(self.size)) # biases of the current nodes
        self.weights = numpy.random.normal(loc=0, scale=0.2, size=(self.size, self.inputs))

    def forward(self, input_values):
        # multiply input_values by weights, then add the bias
        #print(f"size (nodes in current layer): {self.size}")
        #print(f"inputs (nodes in previous layer): {self.inputs}")
        #print(self.weights)
        input_values = numpy.array(input_values)
        if len(input_values.shape) > 1:
            batched_output = []
            for inputs in input_values:
                multiplied_bias = numpy.multiply(self.weights, inputs)
                weighted_nodes = numpy.sum(multiplied_bias, axis=1)
                biased_nodes = numpy.add(weighted_nodes, self.biases)
                batched_output.append(biased_nodes)
            self.output = numpy.array(batched_output)

        else:
            multiplied_biases = numpy.multiply(self.weights, input_values) # same shape and size as self.weights
            #print(f"multiplied biases: {multiplied_biases}")
            weighted_nodes = numpy.sum(multiplied_biases, axis=1) # vector with length of how many nodes are in the current layer
            #print(f"weighted_node: {weighted_nodes}")
            biased_nodes = numpy.add(weighted_nodes, self.biases)
            self.output = biased_nodes
        #print(self.output)

class Activation_Relu:
    def activation_relu(self, inputs):
        if len(inputs.shape) > 1:
            batched_output = []
            for input in inputs:
                outputs = []
                for value in input:
                    if value > 0:
                        outputs.append(value)
                    else:
                        outputs.append(0)
                batched_output.append(outputs)
            self.output = numpy.array(batched_output)

        else:
            outputs = []
            for i in inputs:
                if i > 0:
                    outputs.append(i)
                else:
                    outputs.append(0)
            self.output = outputs

class Activation_Softmax:
    def activation_softmax(self, inputs):
        if len(inputs.shape) > 1:
            batched_output = []
            for input in inputs:
                input = numpy.subtract(input, max(input))
                input = numpy.exp(input)
                summed = numpy.sum(input)
                outputs = []
                for i in input:
                    outputs.append(i/summed)
                batched_output.append(outputs)
            self.output = numpy.array(batched_output)

        else:
            inputs = numpy.subtract(inputs, max(inputs))
            inputs = numpy.exp(inputs)
            summed = numpy.sum(inputs)
            outputs = []
            for i in inputs:
                outputs.append(i/summed)
            self.output = numpy.array(outputs)

class Loss_CategoricalCrossEntropy:
    def calculate_loss(self, outputs, labels):
        new = zip(outputs, labels)
        batched_loss = []
        for i in new:
            index = numpy.where(i[1]==1)[0]
            loss = -numpy.log(i[0][index])
            batched_loss.append(loss)
        self.output = numpy.array(batched_loss)



layer_1 = Layer(3, 4)
layer_2 = Layer(4, 4)
layer_3 = Layer(4, 3)

ReLu = Activation_Relu()
Softmax = Activation_Softmax()

Loss = Loss_CategoricalCrossEntropy()

layer_1.forward(inputs)
ReLu.activation_relu(layer_1.output)
a = ReLu.output

layer_2.forward(ReLu.output)
ReLu.activation_relu(layer_2.output)
b = ReLu.output

layer_3.forward(ReLu.output)
Softmax.activation_softmax(layer_3.output)
c = Softmax.output

Loss.calculate_loss(c, labels)
l = Loss.output


print(f"\nINPUTS:\n {inputs}")
print(f"\nOUTPUTS:\n {c}")
print(f"\nLABELS:\n {labels}\n")
print(f"\nCATEGORICAL LOSS:\n {l}\n")




import numpy
import random

class node:
    def __init__(self, type, input_nodes=[]):
        self.type = type
        self.input_nodes = input_nodes
        self.inputs = []
        for node in self.input_nodes:
            self.inputs.append(node.value)

        if self.type == "input":
            self.value = random.randint(0, 20)
        if self.type != "input":
            self.value = 0

        self.no_of_inputs = int(len(self.input_nodes))

        self.bias = numpy.random.normal(0, 1)
        self.input_weights = []
        for i in range(0, self.no_of_inputs):
            weight = numpy.random.normal(0, 1)
            self.input_weights.append(weight)

    def __repr__(self):
        return self.type


    def evaluate(self):
        if self.type == "input":
            print(self.value)
        else:
            self.value = numpy.dot(self.input_weights, self.inputs) + self.bias
            print(self.value)

def run(input_nodes, hidden_nodes, output_nodes):

    for node in hidden_nodes:
        node.evaluate()
    print("OUTPUTS")
    for node in output_nodes:
        node.evaluate()









# INSTANCES

input_node_1 = node("input")
input_node_2 = node("input")

input_nodes = [input_node_1, input_node_2]

hidden_node_1 = node("hidden", input_nodes)
hidden_node_2 = node("hidden", input_nodes)
hidden_node_3 = node("hidden", input_nodes)

hidden_nodes = [hidden_node_1, hidden_node_2, hidden_node_3]

output_node_1 = node("output", hidden_nodes)
output_node_2 = node("output", hidden_nodes)

output_nodes = [output_node_1, output_node_2]




run(input_nodes, hidden_nodes, output_nodes)
# backpropagate for just one neuron, but with two weights:
# This seems to have the problem of exploding gradients.

import numpy

inputs = [1, 2]
weights = numpy.array([0.5, 0.1])
y = 2







def go(weights):
    iw_1 = inputs[0] * weights[0]
    iw_2 = inputs[1] * weights[1]
    activation = iw_1 + iw_2
    #cost = (y - activation) ** 2
    da_dw1 = inputs[0]
    #print(f"da_dw1: {da_dw1}")
    da_dw2 = inputs[1]
    dc_da = 2 * (y - activation)
    #print(f"dc_da: {dc_da}")
    dc_dw1 = da_dw1 * dc_da
    #print(f"dc_dw1: {dc_dw1}")
    dc_dw2 = da_dw2 * dc_da
    return numpy.array([dc_dw1, dc_dw2])



learning_rate = 0.00000000001
x = 0
while x < 10000:
    changes = go(weights)

    weights = weights - (learning_rate * changes)
    x += 1
    print(f"\nX: {x}")
    print(f"CHANGE: {changes}")
    print(f"NEW WEIGHT: {weights}")







    print("\n")
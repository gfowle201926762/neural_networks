# backpropagate just for one neuron with one input and one weight:

import numpy as np

input = 2
weight = 0.5
desired_activation = 2

def go(weight):
    activation = input * weight
    #cost = (activation - desired_activation) ** 2
    da_dw = input
    dc_da = 2 * (activation - desired_activation)
    dc_dw = da_dw * dc_da
    return dc_dw


learning_rate = 0.1
x = 0
while x < 20:
    change = go(weight)
    weight = weight - (learning_rate * change)
    x += 1

    print(f"\nX: {x}")
    print(f"CHANGE: {change}")
    print(f"NEW WEIGHT: {weight}")
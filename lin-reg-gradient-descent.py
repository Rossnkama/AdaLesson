# Importing our libraries
import numpy as np
import matplotlib.pyplot as plt
import random

# Our datasets
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = float(random.randint(0, 10))

# Forward propagation in our computational graph
def feed_forward(x):
    return x * w

# Loss function
def calculate_loss(x, y):
    return (feed_forward(x) - y)**2

def compute_gradient(x, y): # d_loss/d_w
    
    # Notice how we have to manually derive the gradient in our code
    # Not good :(
    return 2 * x * (x * w - y) 
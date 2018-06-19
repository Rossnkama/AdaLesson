# Importing our libraries
import numpy as np
import matplotlib.pyplot as plt
import random

# Our data sets
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


# What
print("Before training, 4 hours spent studying is predicted to get {} points"
      .format(feed_forward(4)))

# Training the AI
alpha = 0.01

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        gradient = compute_gradient(x, y)
        
        # Updating weight
        w = w - 0.01 * gradient # Experiment with hyper-parameter alpha
        loss = calculate_loss(x, y)
        print("x: {}, y: {}, Gradient:{}".format(x, y, gradient))
    print("Epoch:{}, Weight: {}, Loss: {}"
          .format(epoch, w, loss))

print("After training, 4 hours spend studying is predicted to get {} points where 4 hours really got 8 points"
      .format(feed_forward(4)))

# Linear regression without PyTorch

w = 1.0 # We would usually initialise a random weight

# Forward propagation in our computational graph
def feed_forward(x):
    return x * w

# Loss function
def calculate_loss(x, y):
    return (feed_forward(x) - y)**2
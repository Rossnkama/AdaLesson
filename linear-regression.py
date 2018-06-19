# Importing our libraries
import numpy as np
import matplotlib.pyplot as plt

# Our datasets
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# Forward propagation in our computational graph
def feed_forward(x):
    return x * w

# Loss function
def calculate_loss(x, y):
    return (feed_forward(x) - y)**2

# To plot an error graph later
all_weights = []
all_mses = [] # Mean squared errors

for w in np.arange(0.0, 4.1, 0.1):
    
    print('W=', w) # Show the weight
    sum_of_all_loss = 0

    for x, y in zip(x_data, y_data):
        hypothesis_x = feed_forward(x) # This is 
        loss = calculate_loss(x, y)
        sum_of_all_loss += loss
        print("x:", x)
        print("y:", y)
        print("Our hypothesis of x (y):", hypothesis_x)
        print("Our loss/error squared for this weight {}:".format(w), loss)
    
    print("MSE:", loss/3)
    all_weights.append(w)
    all_mses.append(loss/3)

# Plotting graph of how weights effect the loss
plt.title("Loss vs Weights")
plt.plot(all_weights, all_mses)
plt.ylabel('Loss')
plt.xlabel('Weights')
plt.show()
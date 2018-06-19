import torch
from torch.autograd import Variable

'''
    torch.autograd is the module which allows to automatically
    compute gradients via forward + back propagation.

    The Variable class is used to add operational gates to 
    pytorch's computational graph
'''

# Our data sets
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

'''
    Requires grad means you want a gate + gradient added to the
    computational graph everytime w is referenced in code
'''
w = Variable(torch.Tensor([1.0]), requires_grad=True)

def feed_forward(x):
    '''
        The return value is added to computational graph as w is a PyTorch
        variable
    '''
    return x * w 

def compute_loss(x, y):
    '''
        This function's return is added to the computational graph as it's 
        invoking the forward function which uses PyTorch Variable w
    '''
    return (feed_forward(x) - y)**2

# Showing feed forward network performance before 
print("Before training, 4 hours spent studying is predicted to get {} points"
      .format(feed_forward(4)))
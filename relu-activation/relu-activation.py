import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    if isinstance(x, float):
        return max(0, x)
    else:
        x = np.asarray(x)
        return np.maximum(0, x)
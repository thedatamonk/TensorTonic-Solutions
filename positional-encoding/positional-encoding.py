import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    # create a vector of dimension 
    pe = np.zeros((seq_len, d_model))

    # compute numerator - pos - [[0], [1], [2], [3], [4]]
    pos = np.arange(0, seq_len).reshape(-1, 1)

    # compute denominator - i - [0, 2, 4, 6, 8, 10, ... all even indices]
    i = np.arange(0, d_model, 2)
    denom = 1 / (base**(i/d_model))

    # angle needs to be computed for each position and dimension
    angle = pos * denom

    # now for pe even pos, we will use sin
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle[:, :d_model//2])

    return pe
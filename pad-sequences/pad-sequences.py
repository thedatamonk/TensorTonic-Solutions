import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not max_len:
        max_len = max([len(seq) for seq in seqs])

    for i in range(len(seqs)):
        if len(seqs[i]) > max_len:
            seqs[i] = seqs[i][:max_len]
        else:
            seqs[i] = seqs[i] + [pad_value] * (max_len - len(seqs[i]))

    return seqs
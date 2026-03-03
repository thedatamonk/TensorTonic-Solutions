import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    if isinstance(v, list):
        v = np.array(v)
    if isinstance(w, list):
        w = np.array(w)


    mod_w = np.linalg.norm(w)
    mod_v = np.linalg.norm(v)

    if mod_w == 0 or mod_v == 0:
        return np.nan
    
    cos_value = np.clip(np.dot(w, v) / (mod_w * mod_v), -1, 1)
    return np.arccos(cos_value)

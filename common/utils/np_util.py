import numpy as np


class np_random(object):
    """Context manager for numpy random state"""

    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)
        return self.state

    def __exit__(self, exc_type, exc_val, exc_tb):
        np.random.set_state(self.state)

import numpy as np

def condiciones(temp_range=(0, 100)):

    return {
        'A': np.random.uniform(*temp_range),
        'B': np.random.uniform(*temp_range),
        'C': np.random.uniform(*temp_range),
        'D': np.random.uniform(*temp_range)
    }


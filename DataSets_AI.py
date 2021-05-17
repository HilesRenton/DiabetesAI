import numpy as np
feature_set = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
labels = np.array([[1, 0, 0, 1, 1]])
labels = labels.reshape(5, 1)

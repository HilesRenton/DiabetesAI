single_point = np.array([1, 0, 0])
result = sigmoid(np.dot(single_point, weights) + bias)
print(result)

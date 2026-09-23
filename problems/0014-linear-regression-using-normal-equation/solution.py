import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	data_x = np.array(X)
	data_y = np.array(y)

	theta = np.linalg.inv(data_x.T @ data_x) @ data_x.T @ data_y

	return theta
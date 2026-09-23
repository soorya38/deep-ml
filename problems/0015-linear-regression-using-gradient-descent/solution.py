import numpy as np

def linear_regression_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    iterations: int
) -> np.ndarray:

    m, n = X.shape

    y = y.reshape(-1, 1)
    theta = np.zeros((n, 1))

    for _ in range(iterations):
        # 1. Predictions
        predictions = X @ theta

        # 2. Errors / residuals
        error = predictions - y

        # 3. Gradient
        grad = (X.T @ error) / m

        # 4. Update parameters
        theta = theta - alpha * grad

    return theta.flatten()
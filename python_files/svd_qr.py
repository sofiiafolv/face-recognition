import numpy as np


def simultaneous_power_iteration(A, k):
    n, m = A.shape
    Q = np.random.rand(n, k)
    Q, _ = np.linalg.qr(Q)
    Q_prev = Q
    for _ in range(1000):
        Z = A.dot(Q)
        Q, R = np.linalg.qr(Z)
        err = ((Q - Q_prev) ** 2).sum()
        Q_prev = Q
        if err < 1e-3:
            break
    return np.diag(R), Q


def reduced_svd_using_qr(A, k):
    eigenvalues, V = simultaneous_power_iteration(A.T @ A, k)
    sigma = np.sqrt(eigenvalues)
    idx = np.argsort(eigenvalues)[::-1]
    sigma = sigma[idx]
    rank_of_A = np.linalg.matrix_rank(A)
    V = V[:, idx]
    sigma = sigma[:rank_of_A]
    V = V[:, :rank_of_A]
    U = (A @ V) @ np.linalg.inv(np.diag(sigma))
    return U, sigma, V.T

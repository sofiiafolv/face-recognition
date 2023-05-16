import numpy as np

def reduced_svd(A):
  '''
  Input: any matrix A
  Returns: tuple of matrix U, array of singular values and matrix V.T 
  '''
  # your code here
  rank_of_A = np.linalg.matrix_rank(A)

  symmetric_matrix_V = A.T @ A

  eigenvalues_V, eigenvectors_V = np.linalg.eigh(symmetric_matrix_V)
  eigenvalues_V[eigenvalues_V < 0] = 0

  idx = np.argsort(eigenvalues_V)[::-1]   
  V = eigenvectors_V[:,idx]
  V = V[:,:rank_of_A]
  
  sigma = np.sort(np.sqrt(eigenvalues_V))[::-1]
  sigma = sigma[:rank_of_A]
  U = (A @ V) @ np.linalg.inv(np.diag(sigma))
  return U, sigma, V.T


def k_rank_approximation(U, sigma, VT, k):
  return U[:,:k], sigma[:k], VT[:k,:]


def k_rank_approximation_from_scratch(A, k):
  U, sigma, VT = reduced_svd(A)
  return U[:,:k], sigma[:k], VT[:k,:]

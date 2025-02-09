from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

import pickle
import os
import numpy as np
import pandas as pd
import time
import logging
import argparse

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from scipy.spatial import cKDTree

import itertools
import scipy as sp
from scipy.special import factorial, loggamma

## Utilities

class Logger:
    def __init__(self, log_file='app.log', log_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_info(self, message):
        self.logger.info(message)
    
    def log_warning(self, message):
        self.logger.warning(message)
    
    def log_error(self, message):
        self.logger.error(message)
    
    def log_debug(self, message):
        self.logger.debug(message)
    
    def log_critical(self, message):
        self.logger.critical(message)

def save_data(data, folder_path, filename):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def find_closest_cfs(cfs, queries, weights=None):
    if weights is not None:
        W_root = np.sqrt(np.diag(weights))
        weighted_cfs = cfs @ W_root
        weighted_queries = queries @ W_root
    else:
        weighted_cfs = cfs
        weighted_queries = queries
    # Create a KD-tree for points_b
    tree = cKDTree(weighted_cfs)
    # Find the closest point in points_b for each point in points_a
    distances, indices = tree.query(weighted_queries)
    # Get the closest points
    closest_points = cfs[indices]
    return closest_points, distances, np.array(indices)

def find_closest_cfs_with_immutables(cfs, queries, h):
    h = h.astype(bool)
    indices, distances, closest_points = [], [], []
    for q_id, q in enumerate(queries):
      matching_cf_ids = []
      for cf_id, cf in enumerate(cfs):
        if len(h.shape)==2 and (cf[h[q_id]] == q[h[q_id]]).all():
          matching_cf_ids.append(cf_id)
        elif len(h.shape)==1 and (cf[h] == q[h]).all():
          matching_cf_ids.append(cf_id)

      if len(matching_cf_ids) == 0:
        indices.append(None)
        distances.append(-1)
        closest_points.append(-1*np.ones_like(cfs[0]))
        continue

      matching_cfs = cfs[matching_cf_ids]
      tree = cKDTree(matching_cfs)
      # Find the closest point in points_b for each point in points_a
      distance, index = tree.query(q)
      index = matching_cf_ids[index]

      # Get the closest points
      closest_point = cfs[index]
      indices.append(index)
      distances.append(distance)
      closest_points.append(closest_point)
    distances = np.array(distances)
    indices = np.array(indices)
    closest_points = np.array(closest_points)
    return closest_points, distances, indices

def draw_samples(arr, n, replace=False):
    if replace:
        # Use np.random.choice for sampling with replacement
        indices = np.random.choice(len(arr), size=n, replace=True)
    else:
        # Use np.random.choice with replace=False for sampling without replacement
        indices = np.random.choice(len(arr), size=n, replace=False)
    return arr[indices]

def mod_inverse(a, m):
    """Compute the modular multiplicative inverse of a modulo m."""
    def egcd(a, b):
        if a == 0:
            return (b, 0, 1)
        else:
            g, y, x = egcd(b % a, a)
            return (g, x - (b // a) * y, y)

    g, x, _ = egcd(a, m)
    if g != 1:
        raise Exception('Modular inverse does not exist')
    else:
        return x % m

def matrix_inverse_2x2_finite_field(matrix, q):
    """
    Compute the inverse of a 2x2 matrix in GF(q).

    Parameters:
    matrix (numpy.ndarray): 2x2 matrix to invert
    q (int): Prime number defining the finite field GF(q)

    Returns:
    numpy.ndarray: Inverted 2x2 matrix in GF(q)
    """
    a, b, c, d = matrix.flatten()

    # Compute determinant
    det = (a * d - b * c) % q

    # Check if determinant is invertible
    try:
        det_inv = mod_inverse(det, q)
    except Exception:
        raise ValueError("Matrix is not invertible in GF({})".format(q))

    # Compute adjugate matrix
    adj = np.array([[d, -b], [-c, a]], dtype=int)

    # Compute inverse
    inv = (det_inv * adj) % q

    return inv

def matrix_inverse_3x3_finite_field(matrix, q):
    """
    Compute the inverse of a 3x3 matrix in GF(q).

    Parameters:
    matrix (list of lists): A 3x3 matrix represented as a list of lists
    q (int): The order of the finite field (must be prime)

    Returns:
    list of lists: The inverse matrix, or None if the matrix is not invertible
    """
    def mod_inverse(a, m):
        """Helper function to find modular multiplicative inverse"""
        for i in range(1, m):
            if (a * i) % m == 1:
                return i
        return None

    # Compute the determinant
    det = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
           - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
           + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])) % q

    # Check if the matrix is invertible
    if det == 0:
        return None

    # Compute the modular multiplicative inverse of the determinant
    det_inv = mod_inverse(det, q)
    if det_inv is None:
        return None

    # Compute the adjugate matrix
    adj = [
        [(matrix[1][1]*matrix[2][2] - matrix[1][2]*matrix[2][1]) % q,
         (matrix[0][2]*matrix[2][1] - matrix[0][1]*matrix[2][2]) % q,
         (matrix[0][1]*matrix[1][2] - matrix[0][2]*matrix[1][1]) % q],
        [(matrix[1][2]*matrix[2][0] - matrix[1][0]*matrix[2][2]) % q,
         (matrix[0][0]*matrix[2][2] - matrix[0][2]*matrix[2][0]) % q,
         (matrix[0][2]*matrix[1][0] - matrix[0][0]*matrix[1][2]) % q],
        [(matrix[1][0]*matrix[2][1] - matrix[1][1]*matrix[2][0]) % q,
         (matrix[0][1]*matrix[2][0] - matrix[0][0]*matrix[2][1]) % q,
         (matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]) % q]
    ]

    # Multiply the adjugate matrix by the inverse of the determinant
    inv = [[((det_inv * adj[i][j]) % q) for j in range(3)] for i in range(3)]

    return np.array(inv)

def unique_vectors(vectors):
    """Get unique vectors from a list of vectors."""
    # Convert the list of vectors to a 2D numpy array
    arr = np.array(vectors)
    # Use numpy.unique to get unique rows
    unique_arr = np.unique(arr, axis=0)
    return unique_arr

def compute_cf_accuracy_with_ids(db, x, q_cf_ids, raw_cf_ids):
  cf1 = db[q_cf_ids]
  cf2 = db[raw_cf_ids]
  n_correct = np.linalg.norm(cf1-x, ord=2, axis=1) <= np.linalg.norm(cf2-x, ord=2, axis=1)
  return np.mean(n_correct)

def compute_cf_accuracy(x, q_cfs, raw_cfs):
  n_correct = np.linalg.norm(q_cfs-x, ord=2, axis=1) <= np.linalg.norm(raw_cfs-x, ord=2, axis=1)
  return np.mean(n_correct)

def next_prime(n):
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    next_num = n + 1
    while True:
        if is_prime(next_num):
            return next_num
        next_num += 1

def same_order_largest_to_smallest(arr1, arr2):
    """
    Check if the order of elements from largest to smallest is the same in both arrays.

    Parameters:
    arr1 (list): First array
    arr2 (list): Second array

    Returns:
    bool: True if the order from largest to smallest is the same, False otherwise
    """
    if len(arr1) != len(arr2):
        return False

    # Create sorted indices for both arrays
    sorted_indices1 = sorted(range(len(arr1)), key=lambda k: arr1[k], reverse=True)
    sorted_indices2 = sorted(range(len(arr2)), key=lambda k: arr2[k], reverse=True)

    # Compare the sorted indices
    return sorted_indices1 == sorted_indices2

def normalize_array(arr, method='min-max', axis=0):
    """
    Normalize a 2D NumPy array column-wise.

    Parameters:
    arr (numpy.ndarray): The input 2D array
    method (str): Normalization method. Options are 'min-max', 'z-score', 'l1', 'l2'
    axis (int): The axis along which to normalize (0 for columns, 1 for rows)

    Returns:
    numpy.ndarray: Normalized array
    dict: Dictionary containing normalization parameters for each column
    """

    if arr.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array")

    normalized_arr = np.zeros_like(arr, dtype=float)
    normalization_params = {}

    if method == 'min-max':
        min_vals = np.min(arr, axis=axis, keepdims=True)
        max_vals = np.max(arr, axis=axis, keepdims=True)
        normalized_arr = (arr - min_vals) / (max_vals - min_vals)
        normalization_params = {'min': min_vals, 'max': max_vals}

    elif method == 'z-score':
        mean = np.mean(arr, axis=axis, keepdims=True)
        std = np.std(arr, axis=axis, keepdims=True)
        normalized_arr = (arr - mean) / std
        normalization_params = {'mean': mean, 'std': std}

    elif method == 'l1':
        l1_norm = np.sum(np.abs(arr), axis=axis, keepdims=True)
        normalized_arr = arr / l1_norm
        normalization_params = {'l1_norm': l1_norm}

    elif method == 'l2':
        l2_norm = np.sqrt(np.sum(arr**2, axis=axis, keepdims=True))
        normalized_arr = arr / l2_norm
        normalization_params = {'l2_norm': l2_norm}

    else:
        raise ValueError("Invalid normalization method. Choose 'min-max', 'z-score', 'l1', or 'l2'.")

    return normalized_arr, normalization_params

def nPr_log(n, r):
    return np.exp(loggamma(n+1) - loggamma(n-r+1))

def generate_binary_vectors(length, weight):
    """Generates all binary vectors of specified length and Hamming weight."""
    vectors = []
    for positions in itertools.combinations(range(length), weight):
        vector = [0] * length
        for pos in positions:
            vector[pos] = 1
        vectors.append(vector)
    return np.array(vectors)

def unique_row_ids(arr):
    """
    Returns the indices of unique rows in a NumPy array.

    Args:
        arr (np.ndarray): The input NumPy array.

    Returns:
        np.ndarray: An array of indices corresponding to the unique rows in the input array.
    """
    # Convert the array to a structured array for comparison of rows
    struct_arr = arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    
    # Find unique rows and their indices
    _, unique_indices = np.unique(struct_arr, return_index=True)
    
    return unique_indices


## Quantization scheme

class QuantizationScheme:
  def __init__(self, num_levels, db):
    self.quant_min, self.quant_max, self.step_size = [], [], []
    self.num_levels = num_levels
    self.quant_maps = {}
    for column_id in range(db.shape[1]):
      column = db[:,column_id]
      self.quant_min.append(column.min())
      self.quant_max.append(column.max())
      col_step_size = (self.quant_max[column_id] - self.quant_min[column_id]) / (self.num_levels - 1)
      self.step_size.append(col_step_size)

      quant_map = {}
      for i in range(num_levels):
          lower_bound = column.min() + i * col_step_size
          upper_bound = lower_bound + col_step_size
          symbol = i
          quant_map[symbol] = (lower_bound, upper_bound)
          self.quant_maps[column_id] = quant_map

  def quantize_value(self, value, quant_map):
    #Quantize a single value using the quantization map.
    for symbol, (lower, upper) in quant_map.items():
        if lower <= value < upper:
            return symbol
    return list(quant_map.keys())[-1]

  def quantize(self, dataset):
      # Perform uniform quantization on each column of the dataset.
      quantized_dataset = dataset.copy()
      if dataset.shape[1] != len(self.quant_maps.keys()):
        raise Exception(f'quantization no. of cols mismatch: input cols={dataset.shape[1]}, init cols={len(self.quant_maps.keys())}')
      for column, quant_map in self.quant_maps.items():
        qfunc = lambda x: self.quantize_value(x, quant_map)
        quantized_dataset[:,column] = np.array([qfunc(v) for v in dataset[:,column]])
      return quantized_dataset


## Compute leakage

def compute_leakage(X, y, R, M, scheme, d_min_range, logger:Logger):
    # scheme: 'vanilla' for vanilla, 'diff' for difference, 'mask' for masking
    d = X.shape[1]
    
    if scheme in ('vanilla', 'diff'):
        d_min_range = 2

    # L = R**2*d+1
    # q = next_prime(R**2 * d * L)
    q = next_prime(R**2 * d)

    if d_min_range - 1 > q:
        print(f'WARNING: d_min might exceed q. q={q}, d_min_range={d_min_range}')
        logger.log_warning(f'WARNING: d_min might exceed q. q={q}, d_min_range={d_min_range}')

    quantize = QuantizationScheme(num_levels=R+1, db=X).quantize
    X_quantized = quantize(X)
    unique_ids = unique_row_ids(X_quantized)

    X_quantized = X_quantized[unique_ids]
    y_quantized = y[unique_ids]

    X_accepted = X_quantized[y_quantized==1]
    X_rejected = X_quantized[y_quantized==0]

    # print(f'R={R} d={d} M={M} q={q} scheme={scheme} len_X_accepted={len(X_accepted)} len_X_rejected={len(X_rejected)}')
    logger.log_info(f'R={R} d={d} M={M} q={q} d_min_range={d_min_range} scheme={scheme} len_X_accepted={len(X_accepted)} len_X_rejected={len(X_rejected)}')

    c_entropies, mis = [], []
    for d_min in range(1,d_min_range):
        print(f'd_min={d_min}')
        counts = []
        for x_id in range(len(X_rejected)):
            x = X_rejected[x_id, :]
            cfs = X_accepted
            
            norms = (np.sum((cfs-x)**2, axis=1)) % q
            counts_x = {}
            for db_combination in itertools.permutations(norms, M):
                db_combination = np.array(db_combination)
                for mu_combination in itertools.product(range(d_min), repeat=M):
                    mu_combination = np.array(mu_combination)
                    noisy_combination = tuple((db_combination + mu_combination) % q)

                    if scheme == 'diff':
                        noisy_combination = tuple([noisy_combination[i+1]-noisy_combination[i] for i in range(len(noisy_combination)-1)])
                    
                    if noisy_combination in counts_x.keys():
                        counts_x[noisy_combination] += 1
                    else:
                        counts_x[noisy_combination] = 1
            counts.extend(list(counts_x.values()))

        p_x = 1 / len(X_rejected)
        p_ytuple = 1 / nPr_log(len(X_accepted), M)
        p_mutuple = 1 / d_min**M

        c_entropy = -p_x * p_ytuple * p_mutuple * np.sum([count * np.log(p_ytuple * p_mutuple * count) for count in counts])
        c_entropy = c_entropy/(np.log(q))
        c_entropies.append(float(c_entropy))
        leak = float(c_entropy - M * np.log(d_min)/np.log(q))

        logger.log_info(f'd_min={d_min}, leakage={leak}')
        mis.append(leak)
    return mis





def main(R, M, scheme, d_min_range):
    logger = Logger('./log.txt')
    logger.log_info('************ run start ************')

    dataset_name = 'COMPAS'

    if dataset_name == 'COMPAS':
        dataset = load_dataset("imodels/compas-recidivism")
    logger.log_info(f'Dataset: {dataset_name}')
    # Convert the 'train' split to a pandas DataFrame
    df = pd.DataFrame(dataset['train'])
    # split target and features
    X = df.drop('is_recid', axis=1).to_numpy()
    X = normalize_array(X)[0]
    y = df['is_recid'].to_numpy()

    # R = 2
    # M = 2
    # scheme = 'mask'
    # d_min_range = 3

    leakages = compute_leakage(X, y, R, M, scheme, d_min_range, logger)
    logger.log_info(f'leakages: {leakages}')

    logger.log_info('************ run end ************')


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('R', type=int,  help='A required integer positional argument')
    parser.add_argument('M', type=int,  help='A required integer positional argument')
    parser.add_argument('scheme', type=str,  help='A required integer positional argument')
    parser.add_argument('d_min_range', type=int,  help='A required integer positional argument')

    args = parser.parse_args()
    main(args.R, args.M, args.scheme, args.d_min_range)



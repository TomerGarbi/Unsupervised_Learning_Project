import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from sklearn.cluster import KMeans, SpectralClustering as skl_spec
from sklearn.datasets import make_circles, make_moons, make_blobs
import util


# calculate the rbf distance between vector
def RBF(d1, d2, sigma):
    return np.exp(-(np.linalg.norm(d1 - d2))/(2*sigma**2))


# form the k neighbors adjacency matrix from the data
def k_neighbors_adjacency_graph(data, neighbors):
    neighbors = neighbors + 1  # each point include self as nearest neighbor
    N = len(data)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i][j] = np.linalg.norm(data[i] - data[j])
    for i in range(N):
        row = K[i]
        near_index = []
        near_value = []
        j = 0
        while j < len(row):
            if len(near_index) < neighbors:
                near_index.append(j)
                near_value.append(row[j])
            else:
                max = np.argmax(near_value)
                if row[j] < near_value[max]:
                    near_value[max] = row[j]
                    near_index[max] = j
            j += 1
        for j in range(len(row)):
            if j not in near_index:
                row[j] = 0
            else:
                row[j] = 1
    return K


# form the adjacency matrix based on the rbf weight
def simmilarity_matrix(data):
    N = len(data)
    sim = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            weight = RBF(data[i, :], data[j, :], 0.5)
            sim[i][j] = weight
    return sim


# form the degree matrix which holds the sum of each row of adjacency matrix on the diagonal
def degree_matrix(sim):
    D = np.array(sim.sum(axis=1))
    D = np.diag(D)
    return D


# form the laplacian:  L = degree_matrix - adjacency_matrix
def laplacian(sim, deg):
    return deg - sim


# find, sort and arrange the eigen vectors of laplacian
def smallest_k_eigen_vectors(L, k):
    e_vals, e_vecs = np.linalg.eig(L)
    ind = e_vals.real.argsort()[:k]
    result = np.ndarray(shape=(L.shape[0], 0))
    for i in range(1, ind.shape[0]):
        cor_e_vec = np.transpose(np.matrix(e_vecs[:, ind[i].item()]))
        result = np.concatenate((result, cor_e_vec), axis=1)
    return result


# preform the spectral clustering algorithm over the data
# cluster the corresponding eigenvectors using k-means
def spectral_clustering(data, k, method='rbf'):
    N = len(data)
    data = np.array(data)
    if method == 'rbf':
        sim = simmilarity_matrix(data)
    elif method == 'knn':
        sim = k_neighbors_adjacency_graph(data, 10)
    deg = degree_matrix(sim)
    L = laplacian(sim, deg)
    to_cluster = np.real(smallest_k_eigen_vectors(L, k))  # np.real used to avoid complex numbers from round off errors
    p = KMeans(k).fit(to_cluster)
    return p.labels_

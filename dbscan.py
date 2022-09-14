import numpy as np
import matplotlib.pyplot as plt
import queue
from sklearn.datasets import make_circles, make_blobs
from sklearn.cluster import DBSCAN as sklDBS
import util

# preform the DBSCAN algorithm over the data
# returns the labels and the number of centroids
def DBSCAN(data, eps=0.1, min_points=50):
    data_c = np.array(data)
    N = len(data)
    labels = -3 * np.ones(N)
    neighbors_list = []
    # find the type of each point of the data
    for i in range(N):
        p = data_c[i]
        p_list = np.where(np.linalg.norm(p - data_c, axis=1) < eps)[0]
        p_list = p_list[p_list != i]
        neighbors_list.append(p_list)
        count_neighbors = len(p_list)
        if count_neighbors >= min_points:
            labels[i] = -2  # core point
        elif count_neighbors > 0:
            labels[i] = -1  # border point
        else:
            pass  # noise

    # start from core point and find all accessible data points from the core
    # assign the same label for all those data points
    centroid = 0
    for i in range(N):
        q = queue.Queue()
        jump = 0
        if labels[i] == -2:
            labels[i] = centroid
            q.put(i)
            jump = 1
        elif labels[i] == -1:
            labels[i] = centroid
        while not q.empty():
            neighbors = neighbors_list[q.get()]
            for p in neighbors:
                if labels[p] == -2:
                    labels[p] = centroid
                    q.put(p)
                elif labels[p] == -1:
                    labels[p] = centroid
        centroid = centroid + jump

    return labels, centroid

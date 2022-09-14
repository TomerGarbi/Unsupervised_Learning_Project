import numpy as np
import matplotlib.pyplot as plt
import util
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.cluster import AgglomerativeClustering as skl_agg


# merge to clusters
def merge(d1, d2):
    return [d1, d2]


# flattens clusters
def cluster_to_list(c):
    if type(c) != list:
        return [c]
    c0 = c[0]
    c1 = c[1]
    if type(c0) is list:
        l0 = cluster_to_list(c0)
    else:
        l0 = [c0]
    if type(c1) is list:
        l1 = cluster_to_list(c1)
    else:
        l1 = [c1]
    for i in l1:
        l0.append(i)
    return l0


# calcualte the distance between clusters via max points method
def cluster_dist_max_points(c1, c2):
    l1 = cluster_to_list(c1)
    l2 = cluster_to_list(c2)
    max_dist = np.linalg.norm(l1[0] - l2[0])
    for d1 in l1:
        for d2 in l2:
            dist = np.linalg.norm(d1 - d2)
            if dist > max_dist:
                max_dist = dist
    return max_dist


# calculate the distance between clusters using Ward's method
def cluster_dist_ward(c1, c2, m):
    c1 = cluster_to_list(c1)
    c2 = cluster_to_list(c2)
    mean = np.zeros(m)
    counter = 0
    for l in c1:
        mean += l
        counter += 1
    for l in c2:
        mean += l
        counter += 1
    mean = mean / counter
    sum = 0
    for l in c1:
        sum += np.linalg.norm(l - mean)
    for l in c2:
        sum += np.linalg.norm(l - mean)
    return sum


# returns the indices with minimum distance
def min_dist_pair(clusters, method, m):
    min_ind = (0, 1)
    if method == 'ward':
        min_dist = cluster_dist_ward(clusters[0], clusters[1], m)
    else:
        min_dist = cluster_dist_max_points(clusters[0], clusters[1])
    for i in range(len(clusters) - 1):
        for j in range(i+1, len(clusters)):
            if method == 'ward':
                dist = cluster_dist_ward(clusters[i], clusters[j], m)
            else:
                dist = cluster_dist_max_points(clusters[i], clusters[j])
            if dist < min_dist:
                min_dist = dist
                min_ind = (i, j)
    return min_ind


# preform the agglomerative clustering method over th data
# returns the clusters
def Agglomerative_Clustering(data, k, method='max'):
    clusters = []
    for d in data:
        clusters.append(np.array(d))

    while len(clusters) > k:
        c1, c2 = min_dist_pair(clusters, method, 2)
        new_cluster = merge(clusters[c1], clusters[c2])
        del(clusters[c1])
        if c2 > c1:
            c2 = c2 - 1
        del(clusters[c2])
        clusters.append(new_cluster)
    return clusters

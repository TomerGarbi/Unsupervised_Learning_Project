import math
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as skl_kmeans


# generates k random centroids
def random_centroids(k, n,):
    return np.random.uniform(0, 1, (k, n))


# make sure that every gaussian has different mean
def check_if_exist(e, k):
    for g in k:
        if g[0] == e:
            return True
    return False


# generate k random pairs of expectation and variety for the different gaussians
# 0 <= expectation <= 1
# 0.01 <= variety <= 0.1
def generate_k_different_gaussians(k, le, he, lr, hr):
    k_gaussians = []
    for i in range(k):
        e = np.random.uniform(le, he)
        while check_if_exist(e, k_gaussians):
            e = np.random.uniform(le, he)
        l = np.random.uniform(lr, hr)
        k_gaussians.append((e, l))
    return k_gaussians


# create N 2D random vectors from k different gaussians
def random_2D_data(k, N, le=0, he=1, lr=0.1, hr=1):
    labels = np.zeros(N, int)
    each = math.floor(N/k)
    j = 0
    gaussians = generate_k_different_gaussians(k, le, he, lr, hr)
    data = ([[0, 0]])  # just for initializing, will be deleted afterwards
    for i in range(k):
        points = np.random.normal(gaussians[i][0], gaussians[i][1], (each, 2))
        data = np.concatenate((data, points))
        while j < each * (i+1):
            labels[j] = i
            j = j + 1
    i = 0
    data = np.delete(data, 0, axis=0)
    while j < N:
        point = np.random.normal(gaussians[i][0], gaussians[i][1], (1, 2))
        data = np.concatenate((data, point))
        labels[i] = i
        i = i + 1
        j = j + 1
    return data, labels



# plots each group in different color
def plot_2d(labels, data, k, N, title):
    groups = [[] for g in range(k)]
    i = 0
    while i < N:
        label = labels[i]
        groups[label].append(data[i])
        i = i + 1
    i = 0
    while i < len(groups):
        x = []
        y = []
        j = 0
        while j < len(groups[i]):
            point = groups[i][j]
            x.append(point[0])
            y.append(point[1])
            j = j + 1
        plt.plot(x, y, ".")
        i = i + 1
    plt.title(title)
    plt.show()


def plot(data, labels, title):
    groups = []
    ind = 0
    labels_seen = []
    N = len(data)
    while(ind < N):
        label = labels[ind]
        if label in labels_seen:
            groups[labels_seen.index(label)].append(data[ind])
            ind = ind + 1
        else:
            labels_seen.append(label)
            groups.append([])
    plot_groups(groups, title)


def plot_groups(groups, title):
    i = 0
    while i < len(groups):
        x = []
        y = []
        j = 0
        while j < len(groups[i]):
            point = groups[i][j]
            x.append(point[0])
            y.append(point[1])
            j = j + 1
        plt.plot(x, y, ".")
        i = i + 1
    plt.title(title)
    plt.show()


def is_equal(d1, d2):
    for i in range(len(d1)):
        if d1[i] != d2[i]:
            return False
    return True



def cluster_to_list(c):
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



def plot_clusters(clusters, title):
    for c in clusters:
        l = cluster_to_list(c)
        x = []
        y = []
        for p in l:
            x.append(p[0])
            y.append(p[1])
        plt.plot(x, y, '.')
    plt.title(title)
    plt.show()

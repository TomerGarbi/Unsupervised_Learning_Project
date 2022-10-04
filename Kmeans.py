import math
import time

import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as skl_kmeans


# ---- k-means algorithm -----

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
def generate_k_different_gaussians (k):
    k_gaussians = []
    for i in range(k):
        e = np.random.uniform(0, 1)
        while check_if_exist(e, k_gaussians):
            e = np.random.uniform(0, 1)
        l = np.random.uniform(0.01, 0.1)
        k_gaussians.append((e, l))
    return k_gaussians


# create N 2D random vectors from k different gaussians
def random_2D_data(k, N):
    labels = np.zeros(N, int)
    each = math.floor(N/k)
    j = 0
    gaussians = generate_k_different_gaussians(k)
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


# find the new centroids for the data with respect to the new labels
def find_new_centroids(data, labels, k):
    counters = np.zeros(k)
    acc = np.zeros((k, len(data[0])))
    i = 0
    while i < len(data):
        label = labels[i]
        acc[label] += data[i]
        counters[label] += 1
        i = i + 1
    for i in range(k):
        if counters[i] != 0:
            acc[i] = acc[i] / counters[i]
    return acc


# iterates over the data until all vectors find their center
# returns the centroids, labels and number of iterations
def k_means(data, k, MAX_ITER=100, MIN_CHANGES=0):
    labels = np.zeros(len(data), int)
    centroids = random_centroids(k, len(data[0]))
    loop = True
    iter = 0
    while (loop):
        changed = 0
        margins = np.array([centroids - v for v in data])
        norms = np.array([np.linalg.norm(m.T, axis=0) for m in margins])
        new_lables = np.array([np.argmin(n) for n in norms])
        N = len(data)
        for i in range(N):
            if labels[i] != new_lables[i]:
                changed += 1
        labels = new_lables
        centroids = find_new_centroids(data, labels, k)
        loop = changed > MIN_CHANGES and iter < MAX_ITER  # this ables to determine how many changes are sufficient to stop
        iter += 1
    return centroids, labels, iter


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


# create 1000 different 2D vectors, perform the k-means algorithm on the data with k = 1, 5, 1000
# present how different initializing of the centroids may effect the groups
# compare the implementation to the sklearn implementation by present their labels on the same data
def test_2d(k):
    N = 1000
    (data1, true_labels1) = random_2D_data(1, N)
    (data1000, true_labels1000) = random_2D_data(1000, N)
    (data_k, true_labels_k) = random_2D_data(k, N)
    (my_centroids1, my_labels1, my_iter1) = k_means(data1, 1)
    (my_centroids_k1, my_labels_k1, my_iter_k1) = k_means(data_k, k)
    (my_centroids_k2, my_labels_k2, my_iter_k2) = k_means(data_k, k)
    (my_centroids1000, my_labels1000, my_iter1000) = k_means(data1000, 1000)
    skl_res = skl_kmeans(k, random_state=0).fit(data_k)
    plot_2d(my_labels1, data1, 1, N, "Results of the Assignment Implementation with k = 1")
    plot_2d(my_labels1000, data1000, 1000, N, "Results of the Assignment Implementation with k = 1000")
    plot_2d(true_labels_k, data_k, k, N, "True gaussians")
    plot_2d(my_labels_k1, data_k, k, N, "Results of the Assignment Implementation with k = 5")
    plot_2d(my_labels_k2, data_k, k, N, "Results of the Assignment Implementation with k = 5, different centroids")
    plot_2d(skl_res.labels_, data_k, k, N, "Sklearn Implementation")
    print("iteratios for k=", k, ": ")
    print("first run iterations: ", my_iter_k1)
    print("second run iterations: ", my_iter_k2)
    print("sklearn number of iterations: ", skl_res.n_iter_)


# plots the modified image
def plot_image(centroids, labels, title, size):
    mod = []
    d_index = 0
    while d_index < len(labels):
        mod.append(list(centroids[labels[d_index]]))
        d_index += 1
    mod_img = PIL.Image.new("RGB", size)
    i = 0
    pixels = mod_img.load()
    for j in range(size[1]):
        for k in range(size[0]):
            pixels[k, j] = (mod[i][0], mod[i][1], mod[i][2])
            i = i + 1
    plt.imshow(mod_img)
    plt.title(title)
    plt.show()


# perform the k-means algorithm image
# values being tested for k are 3, 9
def mandrill(k, path):
    img = PIL.Image.open(path)
    data = np.array([[0, 0, 0]])
    img = np.array(img)/256
    size = img.shape
    size = (size[0], size[1])
    for i in range(len(img)):
        data = np.concatenate((data, img[i]))
    data = np.delete(data, 0, axis=0)
    (centroids, labels, iter) = k_means(data, k, MIN_CHANGES=150, MAX_ITER=60)
    centroids = np.array(centroids * 256, int)
    title = "Modified Image, k=" + str(k)
    plot_image(centroids, labels, title, size)


def test_mandrill():
    path = "mandrill.png"
    original_image = PIL.Image.open(path)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.show()
    mandrill(3, path)
    mandrill(8, path)


def main():
    test_2d(5)
    test_mandrill()


if __name__ == "__main__":
    main()
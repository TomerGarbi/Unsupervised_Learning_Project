import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles, make_blobs, make_moons, make_gaussian_quantiles
import util
import dbscan
import spectral_clustering
import agglomerative_clustering
from loadMNIST_py import mnist_dataloader
import warnings
warnings.filterwarnings("ignore")

# generate datasets:
N = 400
circles_data, circles_labels = make_circles(N, factor=0.5, noise=0.05)
blobs_data, blobs_labels = make_blobs(N,  random_state=170)
moons_data, moons_labels = make_moons(N, noise=0.05)
X, y = make_blobs(n_samples=N, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
elliptic_blobs_data, elliptic_blobs_labels = (X_aniso, y)
varied_data, varied_labels = make_blobs(n_samples=N, random_state=165)
sic_data = np.loadtxt('test2_data.txt', delimiter=' ')
datasets2 = {'circles': circles_data, 'moons': moons_data, 'in_out': sic_data}
datasets3 = {'blobs: ': blobs_data,'elliptic blobs': elliptic_blobs_data, 'varied blobs': varied_data}
k = 2
# for data in datasets2.items():
#     # KMeans
#     k_means = KMeans(k).fit(data[1])
#     util.plot(data[1], k_means.labels_, "KMeans: " + data[0])
#     # DBSCAN
#     dbscan_labels, number_of_centroids = dbscan.DBSCAN(data[1], eps=0.25, min_points=3)
#     util.plot(data[1], dbscan_labels, "DBSCAN: " + data[0])
#     # Agglomerative clustering
#     agg_clusters = agglomerative_clustering.Agglomerative_Clustering(data[1], k)
#     util.plot_clusters(agg_clusters, "agglomerative: " + data[0])
#     # ward method
#     ward_clusters = agglomerative_clustering.Agglomerative_Clustering(data[1], k, method='ward')
#     util.plot_clusters(ward_clusters, 'ward: ' + data[0])
#     # Spectral clustering(knn)
#     spec_labels = spectral_clustering.spectral_clustering(data[1], k, method='knn')
#     util.plot(data[1], spec_labels, "spectral: " + data[0])
#
# k = 3
# for data in datasets3.items():
#     # KMeans
#     k_means = KMeans(k).fit(data[1])
#     util.plot(data[1], k_means.labels_, "KMeans: " + data[0])
#     # DBSCAN
#     dbscan_labels, number_of_centroids = dbscan.DBSCAN(data[1], eps=0.7, min_points=50)
#     util.plot(data[1], dbscan_labels, "DBSCAN: " + data[0])
#     # Agglomerative clustering
#     agg_clusters = agglomerative_clustering.Agglomerative_Clustering(data[1], k)
#     util.plot_clusters(agg_clusters, "Agglomerative: " + data[0])
#     # ward method
#     ward_clusters = agglomerative_clustering.Agglomerative_Clustering(data[1], k, method='ward')
#     util.plot_clusters(ward_clusters, 'ward: ' + data[0])
#     # Spectral clustering
#     spec_labels = spectral_clustering.spectral_clustering(data[1], k, method='knn')
#     util.plot(data[1], spec_labels, "spectral: " + data[0])
#





def get_groups(data, labels):
    groups = []
    groups_labels = []
    ind = 0
    labels_seen = []
    N = len(data)
    while (ind < N):
        label = labels[ind]
        if label in labels_seen:
            groups[labels_seen.index(label)].append(data[ind])
            groups_labels[labels_seen.index(label)].append(label)
            ind = ind + 1
        else:
            labels_seen.append(label)
            groups.append([])
            groups_labels.append([])
    return groups, groups_labels


def is_equal(x1, x2):
    for i in range(len(x1)):
        if x1[i] != x2[i]:
            return False
    return True


def index_of(A, x):
    for i in range(len(A)):
        if is_equal(A[i], x):
            return i
    return -1


def closest_centroid(centroids, x):
    min_ind = 0
    min_dist = np.linalg.norm(centroids[0] - x)
    for i in range(1, len(centroids)):
        norm = np.linalg.norm(centroids[1] - x)
        if norm < min_dist:
            min_dist = norm
            min_ind = i
    return min_ind


def mean_and_centroid_lables(gr, digits136, digits136_labels):
    means = []
    labels = []
    for g in gr:
        mean = np.zeros(g[0].shape)
        label_counter = np.zeros(10)
        for p in g:
            mean += p
            ind = index_of(digits136, p)
            label_counter[digits136_labels[ind]] += 1
        label = label_counter.argmax()
        mean = mean / len(g)
        means.append(mean)
        labels.append(label)
    return means, labels

def clusters_to_labels(clusters, data):
    labels = np.zeros(len(data))
    label = 0
    for c in clusters:
        l = util.cluster_to_list(c)
        for p in l:
            ind = index_of(data, p)
            labels[ind] = label
        label += 1
    return labels

k = 3
data = mnist_dataloader.load_data()
train_images = data[0][0]
train_labels = data[0][1]
test_images = data[1][0]
test_labels = data[1][1]
digits136 = []
digits136_labels = []
for i in range(500):
    if train_labels[i] in [1, 3, 6]:
        digits136.append(np.array(train_images[i]).flatten())
        digits136_labels.append(train_labels[i])
digits136 = np.array(digits136) / 256


spec_labels = spectral_clustering.spectral_clustering(digits136, k, method='knn')
dbscan_labels = dbscan.DBSCAN(digits136, eps=4, min_points=12)[0]
agglo_labels = clusters_to_labels(agglomerative_clustering.Agglomerative_Clustering(digits136, k), digits136)

spec_groups, spec_groups_labels = get_groups(digits136, spec_labels)
dbscan_groups, dbscan_groups_labels = get_groups(digits136, dbscan_labels)
agglo_groups, agglo_groups_labels = get_groups(digits136, agglo_labels)

spec_means, spec_labels_final = mean_and_centroid_lables(spec_groups, digits136, digits136_labels)
dbscan_means, dbscan_labels_final = mean_and_centroid_lables(dbscan_groups, digits136, digits136_labels)
agglo_means, agglo_labels_final = mean_and_centroid_lables(agglo_groups, digits136, digits136_labels)


test136 = []
test136_labels = []
for i in range(500):
    if test_labels[i] in [1, 3, 6]:
        test136.append(np.array(test_images[i]).flatten())
        test136_labels.append(test_labels[i])


spec_true_counter = 0
dbscan_true_counter = 0
agglo_true_counter = 0
for i in range(len(test136)):
    spec_centroid_ind = closest_centroid(spec_means, test136[i])
    dbscan_centroid_ind = closest_centroid(dbscan_means, test136[i])
    agglo_centroid_ind = closest_centroid(agglo_means, test136[i])
    if test136_labels[i] == spec_labels_final[spec_centroid_ind]:
        spec_true_counter += 1
    if test136_labels[i] == dbscan_labels_final[dbscan_centroid_ind]:
        dbscan_true_counter += 1
    if test136_labels[i] == agglo_labels_final[agglo_centroid_ind]:
        agglo_true_counter += 1

print("spectral clustering accuracy: ", spec_true_counter / len(test136))
print("DBSCAN accuracy", dbscan_true_counter / len(test136))
print("agglomerative clustering accuracy: ", agglo_true_counter / len((test136)))
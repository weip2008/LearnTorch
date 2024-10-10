import torch
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate synthetic data for clustering (e.g., blobs of points)
n_samples = 1000
n_features = 2
n_clusters = 3

data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Convert data to PyTorch tensor
data_tensor = torch.tensor(data, dtype=torch.float)

# K-Means Clustering using Scikit-learn
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(data_tensor)

# Get cluster assignments and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plotting the results
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

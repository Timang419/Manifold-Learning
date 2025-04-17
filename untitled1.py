# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:30:47 2025

@author: akozy
"""
import numpy as np
import matplotlib.pyplot as plt
import umap

import seaborn as sns

# Generate a dataset with clusters
from sklearn.datasets import make_blobs
data, labels = make_blobs(n_samples=300, centers=4, random_state=42)

# Run UMAP step-by-step
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, n_neighbors in enumerate([2, 10, 50]):  
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1)
    embedding = umap_model.fit_transform(data)
    
    ax = axes[i]
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette="viridis", ax=ax)
    ax.set_title(f"UMAP with n_neighbors={n_neighbors}")

plt.show()


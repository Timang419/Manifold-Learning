# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 18:29:56 2025

@author: akozy
"""

# Importing necessary libraries
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names[y]  # The class labels (setosa, versicolor, virginica)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply UMAP
umap_model = umap.UMAP(n_components=2)  # Reduce to 2 dimensions for visualization
X_umap = umap_model.fit_transform(X_scaled)

# Create a DataFrame for easy plotting
import pandas as pd
df_umap = pd.DataFrame(X_umap, columns=['UMAP 1', 'UMAP 2'])
df_umap['species'] = labels

# Plot the UMAP result
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_umap, x='UMAP 1', y='UMAP 2', hue='species', palette='Set2', s=100, alpha=0.7)
plt.title('UMAP Projection of Iris Dataset')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(title='Species')
plt.tight_layout()
plt.show()


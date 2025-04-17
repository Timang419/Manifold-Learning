import yfinance as yf
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set plot style
sns.set(style='whitegrid')


# Define your assets
assets = ['BTC-USD', '^GSPC', 'NVDA']

# Download full data including multi-index
raw_data = yf.download(assets, start='2020-01-01', end='2024-01-01', group_by='ticker', auto_adjust=False)

# Extract Adjusted Close for each asset
adj_close = pd.DataFrame({
    ticker: raw_data[ticker]['Adj Close'] for ticker in assets
})

# Check the result
print(adj_close.head())


# Step 2: Calculate Daily Returns
returns = adj_close.pct_change().dropna()

# Step 3: Standardize the Data
scaler = StandardScaler()
scaled_returns = scaler.fit_transform(returns)

# Step 4: Apply UMAP for Dimensionality Reduction
umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
umap_embeddings = umap_model.fit_transform(scaled_returns)

# Step 5: K-Means Clustering in UMAP Space
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(umap_embeddings)
centroids = kmeans.cluster_centers_

# Step 6: Plot UMAP Projection with Clusters
plt.figure(figsize=(12, 7))
sns.scatterplot(
    x=umap_embeddings[:, 0], y=umap_embeddings[:, 1],
    hue=cluster_labels, palette='Set1', legend='full', alpha=0.7
)


# Plot centroids
plt.scatter(
    centroids[:, 0], centroids[:, 1],
    c='black', s=200, marker='X', label='Centroids'
)

# Enhance plot aesthetics
plt.xlabel('UMAP (Risk-Return Projection 1)', fontsize=12)
plt.ylabel('UMAP (Risk-Return Projection 2)', fontsize=12)
plt.title('UMAP Projection of NVDA, S&P 500, and Bitcoin with K-Means Clustering', fontsize=14)
plt.legend(title='Cluster', fontsize=10)
plt.tight_layout()
plt.show()


# Create a long-form dataframe to track which asset each return row belongs to
returns_long = returns.reset_index()
returns_long['Asset'] = returns.idxmax(axis=1) 

plt.figure(figsize=(12, 7))
sns.scatterplot(
    x=umap_embeddings[:, 0], y=umap_embeddings[:, 1],
    hue=returns_long['Asset'], palette='Dark2', alpha=0.7
)

plt.xlabel('UMAP (Risk-Return Projection 1)', fontsize=12)
plt.ylabel('UMAP (Risk-Return Projection 2)', fontsize=12)
plt.title('UMAP Projection Colored by Dominant Asset', fontsize=14)
plt.legend(title='Asset', fontsize=10)
plt.tight_layout()
plt.show()
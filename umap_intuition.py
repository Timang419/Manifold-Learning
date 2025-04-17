import numpy as np
import matplotlib.pyplot as plt
import umap

# Create a 3D spiral dataset
t = np.linspace(0, 4 * np.pi, 200)
x = t * np.cos(t)
y = t * np.sin(t)
z = t

data = np.vstack((x, y, z)).T  # 3D dataset

# Apply UMAP to reduce to 2D
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1)
embedding = umap_model.fit_transform(data)

# Plot original 3D data
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x, y, z, c=t, cmap='viridis')
ax1.set_title("Original 3D Data")

# Plot UMAP 2D projection
ax2 = fig.add_subplot(122)
ax2.scatter(embedding[:, 0], embedding[:, 1], c=t, cmap='viridis')
ax2.set_title("UMAP Projection to 2D")

plt.show()

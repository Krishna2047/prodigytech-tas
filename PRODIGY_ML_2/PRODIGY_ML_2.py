import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv("D:/house-prices-advanced-regression-techniques/Mall_Customers.csv")

# Preview the first few rows of the dataset
print(data.head())

# Remove unnecessary columns
data_copy = data.copy()  # Create a copy of the original data
data_copy = data_copy.drop(['CustomerID'], axis=1)  # Drop CustomerID column

# One-hot encode the 'Gender' column (if it's categorical)
data_copy = pd.get_dummies(data_copy, drop_first=True)

# Check for missing values
print(data_copy.isnull().sum())

# Normalize the data (Standard Scaling)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_copy)

# View the scaled data
print(data_scaled[:5])

# Elbow method to find the optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Apply K-Means with the chosen number of clusters (e.g., 5)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(data_scaled)  # Get the cluster labels

# Add cluster labels to the original DataFrame (before scaling)
data_copy['Cluster'] = clusters

# Print the first few rows of the dataset with cluster labels
print(data_copy.head())

# Reduce dimensions using PCA to 2D
pca = PCA(n_components=2)
pca_components = pca.fit_transform(data_scaled)

# Create a DataFrame with PCA components and cluster labels
pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = clusters  # Use cluster labels from the kmeans result

# Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segments (K-means Clustering)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# View the cluster centers (on the original scale)
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=data_copy.columns[:-1])  # Remove 'Cluster' column
print(cluster_centers)

# Analyze the statistics of each cluster
for i in range(5):  # Assuming 5 clusters
    print(f"Cluster {i} statistics:")
    print(data_copy[data_copy['Cluster'] == i].describe())
    print("\n")

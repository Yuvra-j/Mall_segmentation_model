from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from models.cluster import create_clusters

def evaluate_clustering(X_scaled):
    inertias = []
    silhouette_scores = []
    for k in range(1, 11):
        kmeans, labels = create_clusters(X_scaled, k)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    plt.plot(range(1, 11), inertias, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()

    plt.plot(range(2, 11), silhouette_scores[1:], 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    plt.show()

    kmeans, labels = create_clusters(X_scaled, 5)
    print(f"Silhouette Score for K=5: {silhouette_score(X_scaled, labels):.4f}")

if __name__ == "__main__":
    print("Evaluation complete")
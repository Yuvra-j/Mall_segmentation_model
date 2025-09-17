from sklearn.cluster import KMeans

def create_clusters(X_scaled, n_clusters=5, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels

if __name__ == "__main__":
    print("Clustering ready")
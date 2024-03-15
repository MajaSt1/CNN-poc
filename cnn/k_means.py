import numpy as np

def k_means(data, k, max_iterations=100):
    # Losowa inicjalizacja centroidów
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]

    for _ in range(max_iterations):
        # Przypisanie punktów do najbliższych centroidów
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)

        # Aktualizacja centroidów
        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]

        # Sprawdzenie warunku stopu
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, clusters

# Przykładowe dane
data = np.array([[1, 2], [2, 3], [8, 7], [9, 8], [5, 7], [5, 5]])

# Wywołanie funkcji k_means
centroids, clusters = k_means(data, k=2)

# Wyświetlenie wyników
print("Centroidy:")
for centroid in centroids:
    print(centroid)

print("\nKlastry:")
for i, cluster in enumerate(clusters):
    print(f"Klaster {i+1}: {cluster}")
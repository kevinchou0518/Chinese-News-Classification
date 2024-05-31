import json
import jieba
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from umap import UMAP
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
vector_size = 10  # Dimensionality of the word vectors
num_data = 50000  # Number of data samples to use

# Load and preprocess the dataset
with open('dataset.json', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]
data = random.sample(data, num_data)  # Randomly sampling data points

texts = [item['title'] for item in data]
true_labels = [item['class'] for item in data]  # Extract actual class labels for evaluation
tokenized_texts = [list(jieba.cut(text)) for text in texts]

# Train a Word2Vec model
model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=5, min_count=1, workers=8)

# Vectorize the text
def vectorize(text, model):
    vector = np.zeros(model.vector_size)
    count = 0
    for word in text:
        if word in model.wv:
            vector += model.wv[word]
            count += 1
    return vector / count if count > 0 else vector

vectors = np.array([vectorize(text, model) for text in tokenized_texts])



# Clustering with NLTK KMeansClusterer
NUM_CLUSTERS = 5
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=1)
assigned_clusters = kclusterer.cluster(vectors, assign_clusters=True)

# Evaluate clustering
ari = adjusted_rand_score(true_labels, assigned_clusters)
nmi = normalized_mutual_info_score(true_labels, assigned_clusters)


print(f'Adjusted Rand Index: {ari:.3f}')
print(f'Normalized Mutual Information: {nmi:.3f}')

# Output the clustering result
for i in range(NUM_CLUSTERS):
    print(f"\nCluster {i}:")
    a, b, c, d, e = 0, 0, 0, 0, 0
    for j, label in enumerate(assigned_clusters):
        if label == i:
            if(true_labels[j] == "政治"):
                a += 1
            elif(true_labels[j] == "體育"):
                b += 1
            elif(true_labels[j] == "財經"):
                c += 1
            elif(true_labels[j] == "遊戲"):
                d += 1
            elif(true_labels[j] == "影劇"):
                e += 1
    print(f"政治: {a}, 體育: {b}, 財經: {c}, 遊戲: {d}, 影劇: {e}")

# UMAP for dimensionality reduction to visualize the clusters
umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
reduced_vectors = umap_model.fit_transform(vectors)

# Plotting the reduced vectors colored by assigned clusters
plt.figure(figsize=(12, 10))
plt.title("2D UMAP Projection of the Clustered Text Data", fontsize=20)
# sns font
sns.set(font_scale=1.2)
sns.scatterplot(
    x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], 
    hue=assigned_clusters, palette=sns.color_palette("hsv", NUM_CLUSTERS),
    legend="full", alpha=0.7,
    s=10,
)
plt.xlabel('UMAP Dimension 1', fontsize=18)
plt.ylabel('UMAP Dimension 2', fontsize=18)
plt.show()
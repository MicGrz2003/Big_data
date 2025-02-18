import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import NMF
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Breast Cancer Dataset')
plt.colorbar(label='Target')
plt.grid(True)
plt.show()

digits = load_digits()
X = digits.data
y = digits.target

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 8))
for i in range(10):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=str(i))
plt.title('t-SNE - Digits Dataset')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Digit')
plt.grid(True)
plt.show()

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

X = lfw_people.data
y = lfw_people.target

n_components = 20 
nmf = NMF(n_components=n_components, random_state=42)
nmf.fit(X)

fig, axes = plt.subplots(4, 5, figsize=(15, 9))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(nmf.components_[i].reshape(lfw_people.images.shape[1:]), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Feature {i+1}')
plt.suptitle('NMF Features - LFW Dataset', fontsize=16)
plt.show()

wine_data = load_wine()
X = wine_data.data
y = wine_data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

svd = TruncatedSVD(n_components=X.shape[1]-1) 
X_svd = svd.fit_transform(X_scaled)

explained_variance_ratio_cumsum = np.cumsum(svd.explained_variance_ratio_)
n_components_optimal = np.argmax(explained_variance_ratio_cumsum >= 0.95) + 1

svd_optimal = TruncatedSVD(n_components=n_components_optimal)
X_svd_optimal = svd_optimal.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
for i, color in zip(range(len(np.unique(y))), ['r', 'g', 'b']):
    plt.scatter(X_svd_optimal[y == i, 0], X_svd_optimal[y == i, 1], label=f'Class {i+1}', c=color)

plt.title('SVD - Wine Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(newsgroups.data)

lda = LatentDirichletAllocation(n_components=20, random_state=42)
normalizer = Normalizer(copy=False)
lda_pipeline = make_pipeline(lda, normalizer)
X_topics = lda_pipeline.fit_transform(X)

most_probable_topic = np.argmax(X_topics, axis=1)

plt.figure(figsize=(10, 6))
plt.hist(most_probable_topic, bins=range(20), alpha=0.75)
plt.title('Przyporządkowanie dokumentów do tematów za pomocą LDA')
plt.xlabel('Temat')
plt.ylabel('Liczba dokumentów')
plt.xticks(range(20))
plt.grid(True)
plt.show()

# Utworzony histogram pokazuje 20 tematów, oraz ilość dokumentów przynależnych do odpowiedniego tematu
# Ewidentnie najwięcej dokumentów jest w temacie 1 i 12

data = pd.read_csv("testy.csv")

X = data.drop(columns=["Test1"])
y = data["Test2"]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

nmf = NMF(n_components=2)
X_nmf = nmf.fit_transform(X)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA')

plt.subplot(1, 3, 2)
plt.scatter(X_nmf[:, 0], X_nmf[:, 1], c=y, cmap='viridis')
plt.title('NMF')

plt.subplot(1, 3, 3)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title('t-SNE')

plt.tight_layout()
plt.show()

# Metoda PCA i NMF są zbliżone do siebie i lekko inne od, t-SNE i wybrałbym właśnie którąś z nich dwóch. 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.cluster import MeanShift, KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.datasets import load_wine
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

iris = load_iris()
X = iris.data
y = iris.target

agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_clustering.fit(X)
agg_labels = agg_clustering.labels_

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=agg_labels, cmap='viridis')
plt.xlabel('Długość płatka')
plt.ylabel('Szerokość płatka')
plt.title('Aglomeracyjne klastrowanie - Iris')
plt.show()

# Gdy dzielimy na dwa klastry, podział jest bardzo wyraźny i prosty do interpretacji, jednak może być zbyt ogólny
# Moim zdaniem lepiej podzielić na 3 klastry gdyż mamy lepszy i bardziej szczegółowy podział jednak 2 z 3 klastrem się zazębiają. 
# Z analizy można też wywnioskować, że im dłuższy płatek tym węższy i na odwrót.  

wine = load_wine()

X = wine.data

gmm = GaussianMixture(n_components=2)
gmm.fit(X)
labels = gmm.predict(X)

plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Mieszane modele rozkładów Gaussa na danych Iris')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')
plt.colorbar(label='Klastry')
plt.show()

# Dla podziału na 3 klastry, Klastr 0 jest dobrze wyodrębniony i zaglomerowany, jednak 1 i 2 są bardzo rozrzucone i się zazębiają. 
# Moim zdaniem lepiej sprawdza się podział na 2 klastry, jednak też nie jest idealnie zagomerowany, oraz jest mniej szczegółowy, gdyz mamy tylko 2 klastry

faces = fetch_olivetti_faces()
X = faces.data

dbscan = DBSCAN(eps=0.1, min_samples=10)
clusters = dbscan.fit_predict(X)

pca = PCA(n_components=3).fit(X)
X_pca = pca.transform(X)

plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title('DBSCAN na danych Iris')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(label='Klastry')
plt.show()

# Wychodzi tylko jeden klaster, nawet dla zmniejszonego eps i zwiększonego min_samples, co może sugerować, że 
# punkty w przestrzeni cech są zbyt oddalone od siebie, co sprawia, że algorytm DBSCAN traktuje większość punktów jako szum lub punkty graniczne.
# Może to też oznaczać, że rzeczywiście nie ma wyraźnych klastrów w danych. 

diabetes = load_diabetes()

X = diabetes.data
linked = linkage(X, 'ward')

plt.figure(figsize=(15, 10))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram Hierarchicznego Klastrowania Iris')
plt.xlabel('Indeksy próbek')
plt.ylabel('Odległość')
plt.show()

# Widac wyrazny podzial na dwie podgrupy pacjentow. Dla podobnych podgrup można stosować podobne środki leczenia i uczyć się z każdej sytuacji i 
# aktualizować model o nowe przypadki z istotnością proporjonalną do odglęgłości między indeksami pacjentów. 

data = pd.read_csv("heart.csv")

X = data.drop(['age', 'sex', 'chol'], axis=1)

X = pd.get_dummies(data.drop(['age', 'sex', 'chol'], axis=1), 
drop_first=True)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

agglomerative = AgglomerativeClustering(n_clusters=2)
agglomerative_labels = agglomerative.fit_predict(X)

dbscan = DBSCAN(eps=3, min_samples=2)
dbscan_labels = dbscan.fit_predict(X)

gmm = GaussianMixture(n_components=2)
gmm.fit(X)
labels = gmm.predict(X)

kmeans_silhouette = silhouette_score(X, kmeans_labels)
agglomerative_silhouette = silhouette_score(X, agglomerative_labels)
dbscan_silhouette = silhouette_score(X, dbscan_labels)
gmm_silhouette = silhouette_score(X, labels)

kmeans_calinski_harabasz = calinski_harabasz_score(X, kmeans_labels)
agglomerative_calinski_harabasz = calinski_harabasz_score(X, agglomerative_labels)
dbscan_calinski_harabasz = calinski_harabasz_score(X, dbscan_labels)
gmm_calinski_harabasz = calinski_harabasz_score(X, labels)

kmeans_davies_bouldin = davies_bouldin_score(X, kmeans_labels)
agglomerative_davies_bouldin = davies_bouldin_score(X, agglomerative_labels)
dbscan_davies_bouldin = davies_bouldin_score(X, dbscan_labels)
gmm_davies_bouldin = davies_bouldin_score(X, labels)

print()
print("Metoda K-means:")
print("Silhouette Score:", kmeans_silhouette)
print("Calinski-Harabasz Score:", kmeans_calinski_harabasz)
print("Davies-Bouldin Score:", kmeans_davies_bouldin)
print()
print("Metoda Agglomerative Clustering:")
print("Silhouette Score:", agglomerative_silhouette)
print("Calinski-Harabasz Score:", agglomerative_calinski_harabasz)
print("Davies-Bouldin Score:", agglomerative_davies_bouldin)
print()
print("Metoda DBSCAN:")
print("Silhouette Score:", dbscan_silhouette)
print("Calinski-Harabasz Score:", dbscan_calinski_harabasz)
print("Davies-Bouldin Score:", dbscan_davies_bouldin)
print()
print("Metoda GMM:")
print("Silhouette Score:", gmm_silhouette)
print("Calinski-Harabasz Score:", gmm_calinski_harabasz)
print("Davies-Bouldin Score:", gmm_davies_bouldin)
print()

# # Współczynnik Silhouette ocenia odległość między klastrami oraz odległość wewnątrz klastrów, dając wartość od -1 
# # do 1, gdzie wartość bliżej 1 oznacza lepsze podziały. Najlepiej wypada dbscan (prawie 0,8), a najgorzej GMM, bo tylko 0,12
# # Indeks Calińskiego-Harabasza mierzy stosunek dyspersji między klastrami do dyspersji wewnątrz klastra, dając większą 
# # wartość dla bardziej zwartej i jednorodnej grupy. Dla dbscan bezkonkurencyjnie wychodzi najlepszy ideks, bo aż 2036,5
# # Indeks Daviesa-Bouldina mierzy odległość między klastrami, gdzie mniejsza wartość oznacza lepsze podziały między klastrami. 
# # Zgodnie z tym zdecydowanie najlepsze podziały uzyskujemy dla metody dbscan - 0,29, a najmniej korzystne dla GMM bo aż 2,6. 
# # Podsumowując zdecydowanie najlepiej wypada dbscan, a najgorzej GMM. Pomiędzy nimi klasuje się (dość podobne co chodzi o jakość wyniku metody)
# # Metoda K-means i Metoda Agglomerative Clustering. 
# # Z punku widzenia medycznego najepsza będzie metoda dbscan, gdyż jest najbardziej dokładna, co jest porządane gdy w grę chodzi zdrowie serca, czyli techncznie nasze życie.
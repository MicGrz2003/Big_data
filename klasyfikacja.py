from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd    
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from io import StringIO
from sklearn.svm import SVC


data = load_iris()
print("Kolumny dostępne w zbiorze danych Iris:")
print(data.feature_names)
print("\nPrzykładowe dane:")
print(data.data[:5])
print("\nEtykiety klas:")
print(data.target_names)
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Dokładność modelu k-NN:", accuracy)
precision = precision_score(y_test, y_pred, average='micro')
print("Precyzja modelu k-NN:", precision)
recall = recall_score(y_test, y_pred, average='micro')
print("Czułość modelu k-NN:", recall)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Macierz pomyłek:", conf_matrix )

mae_0 = mean_absolute_error(y_test, y_pred)

r2_0 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error dla kNN: {mae_0}")
print(f"R^2dla kNN: {r2_0}")


linear_reg = LinearRegression()

linear_reg.fit(X_train, y_train)

y_pred_linear = linear_reg.predict(X_test)

logistic_reg = LogisticRegression()

logistic_reg.fit(X_train, y_train)

y_pred_logistic = logistic_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_linear)

r2 = r2_score(y_test, y_pred_linear)
print(f"Mean Absolute Error dla regresji liniowej: {mae}")
print(f"R^2 dla regresji liniowej: {r2}")

mae_2 = mean_absolute_error(y_test, y_pred_logistic)

r2_2 = r2_score(y_test, y_pred_logistic)
print(f"Mean Absolute Error dla regresji logistycznej: {mae_2}")
print(f"R^2 dla regresji logistycznej: {r2_2}")

# Ciężko wybrac najlepszy model, jednak moim zdaniem najlepszy kompromis między dokładnością i uniwersalnością ma regresja logistyczna, ponieważ 
# ma najmniejszy błąd MAE i najlepszy współczynnik R^2

breast_cancer = load_breast_cancer()

X = breast_cancer.data
y = breast_cancer.target

df = pd.DataFrame(data=X, columns=breast_cancer.feature_names)
df['target'] = y

print("Liczba brakujących wartości w zbiorze danych:")
print(df.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_reg = LogisticRegression()
logistic_reg.fit(X_train_scaled, y_train)
y_pred_logistic = logistic_reg.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred_logistic)
print("Dokładność modelu:", accuracy)

precision = precision_score(y_test, y_pred_logistic, average='micro')
print("Precyzja modelu:", precision)

recall = recall_score(y_test, y_pred_logistic, average='micro')
print("Czułość modelu:", recall)

conf_matrix = confusion_matrix(y_test, y_pred_logistic)
print("Macierz pomyłek:", conf_matrix )

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Dokładność modelu k-NN:", accuracy)

precision = precision_score(y_test, y_pred, average='micro')
print("Precyzja modelu k-NN:", precision)

recall = recall_score(y_test, y_pred, average='micro')
print("Czułość modelu k-NN:", recall)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Macierz pomyłek:", conf_matrix )

# Po kilkukrotnym przetestowaniu wytrenowanych modeli, regresja logistyczna jest najlepszym modelem pod względem dokładnosći, precyzji i czułości 

digits = load_digits()

X = digits.data
y = digits.target

X_flattened = X.reshape(len(X), -1)  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flattened)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_svm)
print("Dokładność modelu:", accuracy)

precision = precision_score(y_test, y_pred_svm, average='micro')
print("Precyzja modelu:", precision)

recall = recall_score(y_test, y_pred_svm, average='micro')
print("Czułość modelu:", recall)

conf_matrix = confusion_matrix(y_test, y_pred_svm)
print("Macierz pomyłek:", conf_matrix )
                                                                                              
# Na przekątnej macierzy pomyłek znajdują się wartości odpowiadające poprawnie sklasyfikowanym cyfrom.
# Wartości poza przekątną reprezentują błędy klasyfikacji, gdzie rzeczywista etykieta różni się od przewidywanej etykiety.
# Na podstawie macierzy pomyłek możemy zidentyfikować, które cyfry są klasyfikowane najczęściej błędnie poprzez analizę wierszy 
# lub kolumn z dużymi wartościami poza przekątną.
# W naszym przypadku pojedyncze wartości są błędnie sklasyfikowane, co widać na wizualizacji.

titanic_data = sns.load_dataset('titanic')
print(titanic_data.head())

print(titanic_data.describe())

titanic_data.hist(figsize=(12, 8))
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='sex', hue='survived', data=titanic_data)
plt.title('Liczba ocalonych i zmarłych w zależności od płci')
plt.show()

titanic_data_cleaned = titanic_data.dropna()
titanic_data_cleaned['class'] = titanic_data_cleaned['class'].astype('category').cat.codes
titanic_data_cleaned['sex'] = titanic_data_cleaned['sex'].astype('category').cat.codes
titanic_data_cleaned['embarked'] = titanic_data_cleaned['embarked'].astype('category').cat.codes
titanic_data_cleaned['who'] = titanic_data_cleaned['who'].astype('category').cat.codes
titanic_data_cleaned['adult_male'] = titanic_data_cleaned['adult_male'].astype('category').cat.codes
titanic_data_cleaned['deck'] = titanic_data_cleaned['deck'].astype('category').cat.codes
titanic_data_cleaned['embark_town'] = titanic_data_cleaned['embark_town'].astype('category').cat.codes
titanic_data_cleaned['alive'] = titanic_data_cleaned['alive'].astype('category').cat.codes
titanic_data_cleaned['alone'] = titanic_data_cleaned['alone'].astype('category').cat.codes
titanic_data_cleaned = pd.get_dummies(titanic_data_cleaned, columns=['sex', 'embarked'], drop_first=True)
print(titanic_data_cleaned.head())

X = titanic_data_cleaned.drop('survived', axis=1)
y = titanic_data_cleaned['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred_lr = logistic_regression.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_lr)
print("Dokładność modelu:", accuracy)

precision = precision_score(y_test, y_pred_lr, average='micro')
print("Precyzja modelu:", precision)

recall = recall_score(y_test, y_pred_lr, average='micro')
print("Czułość modelu:", recall)

conf_matrix = confusion_matrix(y_test, y_pred_lr)
print("Macierz pomyłek:", conf_matrix )

y_pred_lr_proba = logistic_regression.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_lr_proba)
print("ROC-AUC:", roc_auc)
fpr, tpr, _ = roc_curve(y_test, y_pred_lr_proba)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Krzywa ROC (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Krzywa ROC')
plt.legend(loc="lower right")
plt.show()
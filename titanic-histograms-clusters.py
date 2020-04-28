
# This is titanic dataset from Kaggle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_dataset = pd.read_csv('titanic_train.csv')
test_dataset = pd.read_csv('titanic_test.csv')
survived_column_test_dataset = pd.read_csv('gender_submission.csv')

X_survived_column_test = survived_column_test_dataset.iloc[:, 1].values

#-------Training dataset---------------------------

X1 = train_dataset.iloc[:, 1].values
X1 = X1.reshape(-1, 1)

X2 = train_dataset.iloc[:, 3:7].values

X3 = train_dataset.iloc[:, 8].values
X3 = X3.reshape(-1, 1)

X4 = train_dataset.iloc[:, 10].values
X4 = X4.reshape(-1, 1)

y_train = train_dataset.iloc[:, 11].values 
y_train = y_train.reshape(-1, 1)

# Taking care of missing data using Imputer
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
X5 = X2[:, 1]

X5 = X5.reshape(-1, 1)
missingvalues = missingvalues.fit(X5)
X5 = missingvalues.transform(X5)

X6 = X2[:, 0]
X6 = X6.reshape(-1, 1)

X7 = X2[:, 2:4]

X_train = np.concatenate((X1, X6, X5, X7, X3, X4), axis = 1)

X_class = X_train[:, 0]
X_class = X_class.reshape(-1, 1)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

X_embark = X_train[:, 6]
X_embark = X_embark.reshape(-1, 1)

missingvalues1 = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent', verbose = 0)
missingvalues1 = missingvalues1.fit(X_embark)
X_embark = missingvalues1.transform(X_embark)

le = LabelEncoder()
ohe1 = OneHotEncoder()
X_embark = le.fit_transform(X_embark)
X_embark = X_embark.reshape(-1, 1)

X_temp = X_train[:, 1:6]

X_train = np.concatenate((X_class, X_temp, X_embark, y_train), axis = 1)

le1 = LabelEncoder()
X_train[:, 1] = le1.fit_transform(X_train[:, 1])

#------------------Test dataset---------------------------------------------------

X1_test = test_dataset.iloc[:, 1].values
X1_test = X1_test.reshape(-1, 1)

X2_test = test_dataset.iloc[:, 3:7].values

X3_test = test_dataset.iloc[:, 8].values
X3_test = X3_test.reshape(-1, 1)

X4_test = test_dataset.iloc[:, 10].values
X4_test = X4_test.reshape(-1, 1)

# Taking care of missing data with Imputer
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
X5_test = X2_test[:, 1]

X5_test = X5_test.reshape(-1, 1)
missingvalues = missingvalues.fit(X5_test)
X5_test = missingvalues.transform(X5_test)

missingvalues1 = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
X3_test = X3_test.reshape(-1, 1)
missingvalues1 = missingvalues1.fit(X3_test)
X3_test = missingvalues1.transform(X3_test)

X6_test = X2_test[:, 0]
X6_test = X6_test.reshape(-1, 1)

X7_test = X2_test[:, 2:4]

X_test = np.concatenate((X1_test, X6_test, X5_test, X7_test, X3_test, X4_test), axis = 1)

X_test_class = X_test[:, 0]
X_test_class = X_test_class.reshape(-1, 1)

X_test_embark = X_test[:, 6]
X_test_embark = X_test_embark.reshape(-1, 1)

le2 = LabelEncoder()
ohe3 = OneHotEncoder()
X_test_embark = le2.fit_transform(X_test_embark)
X_test_embark = X_test_embark.reshape(-1, 1)

X_test_temp = X_test[:, 1:6]

X_test = np.concatenate((X_test_class, X_test_temp, X_test_embark), axis = 1)

le3 = LabelEncoder()
X_test[:, 1] = le3.fit_transform(X_test[:, 1])

test_result_dataset = pd.read_csv('gender_submission.csv')
y_test = test_result_dataset.iloc[:, 1].values
y_test = y_test.reshape(-1, 1)

X_test = np.concatenate((X_test, y_test), axis = 1)

#---------Combining training and test dataset to create NumPy array of all the variables for all the passengers---------------------------

X = np.concatenate((X_train, X_test), axis = 0)

#-------NumPy array of all the variables for only survivors--------------------------------------
X_survived = []
for i in range(len(X)-1):
    if X[i, 7] == 1:
        X_survived.append(X[i, 0:7])
 
X_survived = np.array(X_survived)       


#----------Applying K-Means Clustering algorithm-----------------------

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_survived)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()  

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_survived)

#------------------------------------------------------------
y_kmeans = y_kmeans.reshape(-1, 1)
X_cluster_survived = np.concatenate((X_survived, y_kmeans), axis = 1)

#-----------Histogram Plots-------------------------------------------------

plt.hist(X_cluster_survived[:, 7])
plt.title('Histogram of Passengers survived')
plt.xlabel('Clusters of Passengers')
plt.ylabel('Number of Passengers survived')
plt.show()

plt.hist(X_cluster_survived[:, 2])
plt.title('Histogram of Passengers survived')
plt.xlabel('Age of Passengers')
plt.ylabel('Number of Passengers survived')
plt.show()

plt.hist(X_cluster_survived[:, 0])
plt.title('Histogram of Passengers survived')
plt.xlabel('Traveling Class of Passengers')
plt.ylabel('Number of Passengers survived')
plt.show()

plt.hist(X_cluster_survived[:, 1])
plt.title('Histogram of Passengers survived')
plt.xlabel('Sex of Passengers, 1 = Male, 0 = Female')
plt.ylabel('Number of Passengers survived')
plt.show()

plt.hist(X_cluster_survived[:, 3])
plt.title('Histogram of Passengers survived')
plt.xlabel('Number of Siblings/Spouses Aboard')
plt.ylabel('Number of Passengers survived')
plt.show()

plt.hist(X_cluster_survived[:, 4])
plt.title('Histogram of Passengers survived')
plt.xlabel('Number of Parents/Children Aboard')
plt.ylabel('Number of Passengers survived')
plt.show()

plt.hist(X_cluster_survived[:, 5])
plt.title('Histogram of Passengers survived')
plt.xlabel('Passenger Ticket Fare')
plt.ylabel('Number of Passengers survived')
plt.show()

plt.hist(X_cluster_survived[:, 6])
plt.title('Histogram of Passengers survived')
plt.xlabel('Port of Embarkation (0 = Cherbourg; 1 = Queenstown; 2 = Southampton)')
plt.ylabel('Number of Passengers survived')
plt.show()

# Visualising the clusters of passengers--------------------------------
y_kmeans = y_kmeans.reshape(-1)
plt.scatter(X_survived[y_kmeans == 0, 5], X_survived[y_kmeans == 0, 2], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_survived[y_kmeans == 1, 5], X_survived[y_kmeans == 1, 2], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_survived[y_kmeans == 2, 5], X_survived[y_kmeans == 2, 2], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_survived[y_kmeans == 3, 5], X_survived[y_kmeans == 3, 2], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 5], kmeans.cluster_centers_[:, 2], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Passengers Survived')
plt.xlabel('Passenger Fare')
plt.ylabel('Age of Passengers')
plt.legend()
plt.show()

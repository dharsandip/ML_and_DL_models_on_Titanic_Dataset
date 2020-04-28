#----This is titanic dataset from Kaggle-------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_dataset = pd.read_csv('titanic_train.csv')

#-------------training dataset------------------------------
X1 = train_dataset.iloc[:, 1].values
X1 = X1.reshape(-1, 1)

X2 = train_dataset.iloc[:, 3:7].values

X3 = train_dataset.iloc[:, 8].values
X3 = X3.reshape(-1, 1)

X4 = train_dataset.iloc[:, 10].values
X4 = X4.reshape(-1, 1)

y_train = train_dataset.iloc[:, 11].values 

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
ohe = OneHotEncoder()
X_class = ohe.fit_transform(X_class).toarray()
X_class = X_class[:, 1:3]

X_embark = X_train[:, 6]
X_embark = X_embark.reshape(-1, 1)

missingvalues1 = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent', verbose = 0)
missingvalues1 = missingvalues1.fit(X_embark)
X_embark = missingvalues1.transform(X_embark)


le = LabelEncoder()
ohe1 = OneHotEncoder()
X_embark = le.fit_transform(X_embark)
X_embark = X_embark.reshape(-1, 1)
X_embark = ohe1.fit_transform(X_embark).toarray()
X_embark = X_embark[:, 1:3]

X_temp = X_train[:, 1:6]

X_train = np.concatenate((X_class, X_temp, X_embark), axis = 1)

le1 = LabelEncoder()
X_train[:, 2] = le1.fit_transform(X_train[:, 2])

#----------------Test dataset--------------------------------------------
#------------------------------------------------------------------------------------

test_dataset = pd.read_csv('titanic_test.csv')

X1_test = test_dataset.iloc[:, 1].values
X1_test = X1_test.reshape(-1, 1)

X2_test = test_dataset.iloc[:, 3:7].values

X3_test = test_dataset.iloc[:, 8].values
X3_test = X3_test.reshape(-1, 1)

X4_test = test_dataset.iloc[:, 10].values
X4_test = X4_test.reshape(-1, 1)

# Taking care of missing data using Imputer
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

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ohe2 = OneHotEncoder()
X_test_class = ohe2.fit_transform(X_test_class).toarray()
X_test_class = X_test_class[:, 1:3]

X_test_embark = X_test[:, 6]
X_test_embark = X_test_embark.reshape(-1, 1)

le2 = LabelEncoder()
ohe3 = OneHotEncoder()
X_test_embark = le2.fit_transform(X_test_embark)
X_test_embark = X_test_embark.reshape(-1, 1)
X_test_embark = ohe3.fit_transform(X_test_embark).toarray()
X_test_embark = X_test_embark[:, 1:3]

X_test_temp = X_test[:, 1:6]

X_test = np.concatenate((X_test_class, X_test_temp, X_test_embark), axis = 1)

le3 = LabelEncoder()
X_test[:, 2] = le3.fit_transform(X_test[:, 2])

#----------Feature scaling----------------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#-------Applying Kernel SVM Classifier----------------------------
from sklearn.svm import SVC
classifier = SVC(C = 1, gamma = 0.2, kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

test_result_dataset = pd.read_csv('gender_submission.csv')
y_test = test_result_dataset.iloc[:, 1].values

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy_of_testset = ((cm[0, 0] + cm[1, 1])/(cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0]))*100


#--------------k-fold cross validation of the training set-----------------------
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
Accuracy_of_trainingset = accuracies.mean()*100
std_of_trainingset = accuracies.std()


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 20, 50, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_



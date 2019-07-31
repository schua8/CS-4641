from sklearn import neighbors
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import cross_val_score, train_test_split, validation_curve, GridSearchCV, learning_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import cross_val_score, validation_curve, GridSearchCV, learning_curve, train_test_split
from functions import plot_learning_curve

bcancer = load_breast_cancer()
training_data, test_data, training_target, test_target = train_test_split(
	bcancer.data, bcancer.target, test_size = .3, random_state = 1)

parameters = {'n_neighbors':list(range(1,10)), 'metric':['euclidean', 'manhattan','minkowski', 'canberra']}
print('Training KNN on Breast Cancer dataset with parameters: ' + str(parameters))
knn = neighbors.KNeighborsClassifier()
clf = GridSearchCV(knn, parameters, cv=5, scoring='f1_weighted')
clf.fit(training_data,training_target)
print(clf.best_params_)
# best params
# num of neighbors - 7
# distance formula - canberra
params = clf.best_params_
clf = neighbors.KNeighborsClassifier(**(params))
clf.fit(training_data,training_target)
print(classification_report(test_target,clf.predict(test_data)))
# score = 0.96 CHECK
title = "Learning Curve for KNN - Breast Cancer"
estimator = neighbors.KNeighborsClassifier(**(params))
plot_learning_curve(estimator, title, training_data, training_target, ylim=(0,1.1),cv=5)
plt.show()

### DIGITS DATASET ###

digits = load_digits()
training_data, test_data, training_target, test_target = train_test_split(
	digits.data, digits.target, test_size = .3, random_state = 1)

parameters = {'n_neighbors':list(range(1,10)), 'metric':['euclidean', 'manhattan','minkowski']}
print('Training KNN on Digits dataset with parameters: ' + str(parameters))
knn = neighbors.KNeighborsClassifier()
clf = GridSearchCV(knn, parameters, cv=5, scoring='accuracy')
clf.fit(training_data,training_target)
print(clf.best_params_)
# best params
# num of neighbors - 3
# distance formula - Euclidean
params = clf.best_params_
clf = neighbors.KNeighborsClassifier(**(params))
clf.fit(training_data,training_target)
print(classification_report(test_target,clf.predict(test_data)))
# score = 0.99 CHECK
title = "Learning Curve for KNN - Digits"
estimator = neighbors.KNeighborsClassifier(**(params))
plot_learning_curve(estimator, title, training_data, training_target, ylim=(0,1.1),cv=5)
plt.show()
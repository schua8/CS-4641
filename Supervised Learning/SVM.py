from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import cross_val_score, validation_curve, GridSearchCV, learning_curve, train_test_split
from functions import plot_learning_curve

### BREAST CANCER DATASET ###

bcancer = load_breast_cancer()
training_data, test_data, training_target, test_target = train_test_split(
	bcancer.data, bcancer.target, test_size = .3, random_state = 1)


scaler = StandardScaler()
scaler.fit(training_data)
training_data = scaler.transform(training_data)
test_data = scaler.transform(test_data)

models = ['linear', 'rbf','poly','sigmoid']
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
parameters = {'kernel':models, 'C':Cs}
vector_mach = svm.SVC(gamma='auto')
print('Training SVM on Breast Cancer dataset with parameters: ' + str(parameters))
clf = GridSearchCV(vector_mach, parameters, cv=5, scoring='f1_weighted')
clf.fit(training_data,training_target)
print(clf.best_params_)
# best params
# C - 1
# kernel - linear
params = clf.best_params_
clf = svm.SVC(gamma='auto',**(params))
clf.fit(training_data,training_target)
print(classification_report(test_target,clf.predict(test_data)))
# score = 0.95 CHECK
title = "Learning Curve for SVM - Breast Cancer"
estimator = svm.SVC(gamma='auto',**(params))
plot_learning_curve(estimator, title, training_data, training_target, ylim=(0,1.1),cv=5)
plt.show()

### DIGITS DATASET ###

digits = load_digits()
training_data, test_data, training_target, test_target = train_test_split(
	digits.data, digits.target, test_size = .3, random_state = 1)


scaler = StandardScaler()
scaler.fit(training_data)
training_data = scaler.transform(training_data)
test_data = scaler.transform(test_data)

models = ['linear', 'rbf','poly','sigmoid']
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
parameters = {'kernel':models, 'C':Cs}
vector_mach = svm.SVC(gamma='auto')
print('Training SVM on Digits dataset with parameters: ' + str(parameters))
clf = GridSearchCV(vector_mach, parameters, cv=5, scoring='accuracy')
clf.fit(training_data,training_target)
print(clf.best_params_)
# best params
# C - 10
# kernel - rbf
params = clf.best_params_
clf = svm.SVC(gamma='auto',**(params))
clf.fit(training_data,training_target)
print(classification_report(test_target,clf.predict(test_data)))
# score = 0.98
title = "Learning Curve for SVM - Digits"
estimator = svm.SVC(gamma='auto',**(params))
plot_learning_curve(estimator, title, training_data, training_target, ylim=(0,1.1),cv=5)
plt.show()
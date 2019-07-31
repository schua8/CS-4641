from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, validation_curve, GridSearchCV, learning_curve, train_test_split
from sklearn.datasets import load_breast_cancer, load_digits
from functions import plot_learning_curve
from time import time
import numpy as np
import matplotlib.pyplot as plt


### BREAST CANCER DATASET ###

bcancer = load_breast_cancer()
training_data, test_data, training_target, test_target = train_test_split(
	bcancer.data, bcancer.target, test_size = .3, random_state = 1)


scaler = StandardScaler()
scaler.fit(training_data)
training_data = scaler.transform(training_data)
test_data = scaler.transform(test_data)

activation = ['identity', 'logistic', 'tanh','relu']
hidden_nodes = [(x,) for x in range(20,140,20)]
parameters = {'activation':activation, 'hidden_layer_sizes':hidden_nodes}
print('Training Neural Net on Breast Cancer dataset with parameters: ' + str(parameters))
nn = MLPClassifier()
clf = GridSearchCV(nn, parameters, cv=5, scoring='f1_weighted')
clf.fit(training_data,training_target)
print(clf.best_params_)
# best params
# activation - tanh
# nodes - 20
params = clf.best_params_
clf = MLPClassifier(**(params))
clf.fit(training_data,training_target)
print(classification_report(test_target,clf.predict(test_data)))
# score = 0.97 CHECK
title = "Learning Curve for Neural Network - Breast Cancer"
estimator = MLPClassifier(**(params))
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

activation = ['identity', 'logistic', 'tanh','relu']
hidden_nodes = [(x,) for x in range(20,140,20)]
parameters = {'activation':activation, 'hidden_layer_sizes':hidden_nodes}
print('Training Neural Net on Digits dataset with parameters: ' + str(parameters))
nn = MLPClassifier()
clf = GridSearchCV(nn, parameters, cv=5, scoring='accuracy')
clf.fit(training_data,training_target)
print(clf.best_params_)
# best params
# activation - relu
# nodes - 80
params = clf.best_params_
clf = MLPClassifier(**(params))
clf.fit(training_data,training_target)
print(classification_report(test_target,clf.predict(test_data)))
# score = 0.98 CHECK
title = "Learning Curve for Neural Network - Digits"
estimator = MLPClassifier(**(params))
plot_learning_curve(estimator, title, training_data, training_target, ylim=(0,1.1),cv=5)
plt.show()
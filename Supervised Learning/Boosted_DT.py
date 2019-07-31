from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import cross_val_score, validation_curve, GridSearchCV, learning_curve, train_test_split
from functions import plot_learning_curve

### BREAST CANCER DATASET ###

bcancer = load_breast_cancer()
training_data, test_data, training_target, test_target = train_test_split(
	bcancer.data, bcancer.target, test_size = .3, random_state = 1)

parameters = {'n_estimators':[25,50,100,150,200], 'min_samples_split':list(range(2,50,2))}
print('Training Boosted Decision Tree on Breast Cancer dataset with parameters: ' + str(parameters))
boosted = GradientBoostingClassifier()
clf = GridSearchCV(boosted, parameters, cv=5, scoring='f1_weighted')
clf.fit(training_data,training_target)
print(clf.best_params_)
# best params
# num of estimators 200
# min split of 34
params = clf.best_params_
clf = GradientBoostingClassifier(**(params))
clf.fit(training_data,training_target)
print(classification_report(test_target,clf.predict(test_data)))
# score = 0.97 CHECK
title = "Learning Curve for Boosted Decision Tree - Breast Cancer"
estimator = GradientBoostingClassifier(**(params))
plot_learning_curve(estimator, title, training_data, training_target, ylim=(0,1.1),cv=5)
plt.show()

### DIGITS DATASET ###

digits = load_digits()
training_data, test_data, training_target, test_target = train_test_split(
	digits.data, digits.target, test_size = .3, random_state = 1)

parameters = {'n_estimators':[25,50,100,200], 'min_samples_split':list(range(2,50,2))}
print('Training Boosted Decision Tree on Digits dataset with parameters: ' + str(parameters))
boosted = GradientBoostingClassifier()
clf = GridSearchCV(boosted, parameters, cv=5, scoring='accuracy')
clf.fit(training_data,training_target)
print(clf.best_params_)
# best params
# num of estimators 100
# min split of 22
params = clf.best_params_
clf = GradientBoostingClassifier(**(params))
clf.fit(training_data,training_target)
print(classification_report(test_target,clf.predict(test_data)))
# score = 0.95 CHECK
title = "Learning Curve for Boosted Decision Tree - Digits"
estimator = GradientBoostingClassifier(**(params))
plot_learning_curve(estimator, title, training_data, training_target, ylim=(0,1.1),cv=5)
plt.show()
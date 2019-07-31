from sklearn.datasets import load_breast_cancer, load_digits
from sklearn import tree
from sklearn.metrics import mean_squared_error, classification_report
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, GridSearchCV, learning_curve
from functions import plot_learning_curve
import matplotlib.pyplot as plt

### BREAST CANCER DATASET ###

bcancer = load_breast_cancer()
training_data, test_data, training_target, test_target = train_test_split(
	bcancer.data, bcancer.target, test_size = .3, random_state = 1)

parameters = {'max_depth': list(range(2,20)), 'min_samples_split':list(range(2,50,2))}
print('Training Decision Tree on Breast Cancer dataset with parameters: ' + str(parameters))
dt = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = GridSearchCV(dt, parameters, cv = 5, scoring = 'f1_weighted') 
clf.fit(training_data, training_target)
print(clf.best_params_)

# Best params:
# max depth of 6
# min split of 18
params = clf.best_params_
clf = tree.DecisionTreeClassifier(criterion = 'entropy', **(params))
clf.fit(training_data, training_target)
print(classification_report(test_target, clf.predict(test_data)))
# Score of 0.91 CHECK

title = "Learning Curve for Decision Tree - Breast Cancer"
estimator = tree.DecisionTreeClassifier(criterion = 'entropy', **(params))
plot_learning_curve(estimator, title, training_data, training_target, ylim = (0,1.1), cv = 5)
plt.show()

#### DIGITS DATASET ####

digits = load_digits()

training_data, test_data, training_target, test_target = train_test_split(
	digits.data, digits.target, test_size = .3, random_state = 1)

parameters = {'max_depth': list(range(2,20)), 'min_samples_split':list(range(2,50,2))}
print('Training Decision Tree on Digits dataset with parameters: ' + str(parameters))
dt = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = GridSearchCV(dt, parameters, cv = 5, scoring = 'accuracy') 
clf.fit(training_data, training_target)
print(clf.best_params_)

# Best params:
# max depth of 15
# min split of 2
params = clf.best_params_
clf = tree.DecisionTreeClassifier(criterion = 'entropy', **(params))
clf.fit(training_data, training_target)
print(classification_report(test_target, clf.predict(test_data)))
# Score of 0.88

title = "Learning Curve for Decision Tree - Digits"
estimator = tree.DecisionTreeClassifier(criterion = 'entropy', **(params))
plot_learning_curve(estimator, title, training_data, training_target, ylim = (0,1.1), cv = 5)
plt.show()
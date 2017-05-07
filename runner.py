import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import gradient_descent
from logistic_regression import log_pred
from random_forest import RandomForest
from decision_tree import DecisionTree


def accuracy_score(Y_true, Y_predict):
    cor_count=0
    for i in range(len(Y_true)):
        if Y_predict[i]==Y_true[i]:
            cor_count+=1
    acc=cor_count/len(Y_true)
    return acc


def evaluate_performance():
    '''
    Evaluate the performance of decision trees and logistic regression,
    average over 1,000 trials of 10-fold cross validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of logistic regression
      stats[1,1] = std deviation of logistic regression accuracy

    ** Note that your implementation must follow this API**
    '''

    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    Y = np.array(data[:, 0])
    n, d = X.shape
    N_folds = 10

    des_tree_acc = []
    rand_forest_acc = []
    log_acc = []

    for trial in range(5):
        print(trial)
        idx = np.arange(n) #shuffling data
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        Y = Y[idx]

        X_train=X[:int(n*0.9),:]
        X_val=X[int(n*0.9):,:]
        Y_train=Y[:int(n*0.9)]
        Y_val=Y[int(n*0.9):]

        # training the decision tree
        des_tree = DecisionTree(100) #create a decision tree with max_depth=100
        des_tree.fit(X_train, Y_train)
        des_tree_pred = des_tree.predict(X_val)
        des_tree_accuracy = accuracy_score(Y_val, des_tree_pred)
        des_tree_acc.append(des_tree_accuracy)

        # training the random forest
        rand_forest = RandomForest(10, 100) # create a random forest with number of trees: 10 and max depth: 100
        rand_forest.fit(X_train, Y_train)
        rand_forest_pred = rand_forest.predict(X_val)[0] #[0] indicates tuple member, because predict returns (Y,conf)
        rand_forest_accuracy = accuracy_score(Y_val, rand_forest_pred)
        rand_forest_acc.append(rand_forest_accuracy)

        #training the logistic regression
        weights = gradient_descent(X_train, Y_train, step_size=1e-3, max_steps=500)
        log_predicted = log_pred(X_val, weights)
        log_accuracy = accuracy_score(Y_val, log_predicted)
        log_acc.append(log_accuracy)

    # compute the training accuracy of the model
    meanDecisionTreeAccuracy = np.mean(des_tree_acc)
    stddevDecisionTreeAccuracy = np.std(des_tree_acc)
    meanLogisticRegressionAccuracy = np.mean(log_acc)
    stddevLogisticRegressionAccuracy = np.std(log_acc)
    meanRandomForestAccuracy = np.mean(rand_forest_acc)
    stddevRandomForestAccuracy = np.std(rand_forest_acc)

    # make certain that the return value matches the API specification
    stats = np.zeros((3, 2))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanRandomForestAccuracy
    stats[1, 1] = stddevRandomForestAccuracy
    stats[2, 0] = meanLogisticRegressionAccuracy
    stats[2, 1] = stddevLogisticRegressionAccuracy
    return stats


# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluate_performance()
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Random Forest Tree Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("Logistic Reg. Accuracy = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.

import numpy as np
import matplotlib.pyplot as plt
import logistic_regression as lr
from decision_tree import DecisionTree
from random_forest import RandomForest

def accuracy_score(Y_true, Y_predict):
    """
    :param Y_true: 2D python list or numpy array
    :param Y_predict: 2D python list or numpy array
    
    :return: integer that represents the accurasy - the percent of correctly predicted labels
    """
    accuracy=0
    for i in range(len(Y_true)):
        if Y_predict[i]==Y_true[i]:
            accuracy+=1
    return accuracy/len(Y_true)


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
    filename = 'SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array(data[:, 0])
    n, d = X.shape
    folds=10

    decision_tree_accuracies=[]
    random_forest_accuracies=[]
    log_regression_accuracies=[]

    for trial in range(3):
        # TODO: shuffle for each of the trials.
        # the following code is for reference only.
        idx = np.arange(n)
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
        size=int(n-n/folds)

        print("trial", trial)

        # TODO: write your own code to split data (for cross validation)
        # the code here is for your reference.
        Xtrain = X[:size,:]# train on first 100 instances
        Xtest = X[size:,:]
        ytrain = y[:size]# test on remaining instances
        ytest = y[size:]

        # train the decision tree
        dt = DecisionTree(100)
        
        dt.fit(Xtrain, ytrain)

        # output predictions on the remaining data
        dt_pred = dt.predict(Xtest)
        dt_accuracy = accuracy_score(ytest, dt_pred)
        decision_tree_accuracies.append(dt_accuracy)

        #train random forest
        rf= RandomForest(10,100)
        rf.fit(Xtrain,ytrain)
        rf_pred= rf.predict(Xtest)[0]
        rf_accuracy= accuracy_score(ytest,rf_pred)
        random_forest_accuracies.append(rf_accuracy)

        #logistic regression
        lr_beta=lr.gradient_descent(Xtrain,ytrain,step_size=1e-1,max_steps=100)

        lr_pred=lr.lr_predict(Xtest,lr_beta)
        lr_accuracy=accuracy_score(ytest,lr_pred)
        log_regression_accuracies.append(lr_accuracy)






    # compute the training accuracy of the model
    meanDecisionTreeAccuracy = np.mean(decision_tree_accuracies)
    stddevDecisionTreeAccuracy = np.std(decision_tree_accuracies)
    meanLogisticRegressionAccuracy = np.mean(log_regression_accuracies)
    stddevLogisticRegressionAccuracy = np.std(log_regression_accuracies)
    meanRandomForestAccuracy = np.mean(random_forest_accuracies)
    stddevRandomForestAccuracy = np.std(random_forest_accuracies)

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
    print ("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    # my results - 67% - Decision tree
    # 78% - Random forest
    print ("Random Forest Tree Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print ("Logistic Reg. Accuracy = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.
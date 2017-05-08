import numpy as np
from dtOmitted import DecisionNode, build_tree

class DecisionTree(object):
    """
    DecisionTree class, that represents one Decision Tree

    :param max_tree_depth: maximum depth for this tree.
    """

    def __init__(self, max_tree_depth):
        self.max_depth = max_tree_depth

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array

        :return: a tree of self.max_tree_depth with helps to classify the data
        """

        if type(X) is np.ndarray:
            X = X.tolist()
        if type(Y) is np.ndarray:
            Y = Y.tolist()

        # In order to get a data corresponding the data structure from decision node, we join X's and Y's in the following way

        data = [X[i] + [Y[i]] for i in range(len(X))]

        self.root = build_tree(data, max_depth=self.max_depth)

    def predict1(self, x, v):

        """
         Predict function for one row from the data X

        :param x: 1 dimensional python list or numpy 1 dimensional array
        :param v: the current node of the tree

        :return: choose and recursively call on the next node based on the values of x
        """

        if v.is_leaf == True:
            return v.result
        else:
            xval = x[v.column]
            val = v.value
            if type(val) == int or type(val) == float:
                if xval > val:
                    return self.predict1(x, v.true_branch)
                else:
                    return self.predict1(x, v.false_branch)
            if type(val) == str:
                if xval == val:
                    return self.predict1(x, v.true_branch)
                else:
                    return self.predict1(x, v.false_branch)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        calls predict1 and returns the final predicted labels for corresponding X's

        :return: Y - 1 dimension python list with labels
        """

        return [self.predict1(row, self.root) for row in X]
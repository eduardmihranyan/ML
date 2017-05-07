import numpy as np
from decision_tree import DecisionTree

class RandomForest(object):
    """
    RandomForest a class, that represents Random Forests.

    :param num_trees: Number of trees in the random forest
    :param max_tree_depth: maximum depth for each of the trees in the forest.
    :param ratio_per_tree: ratio of points to use to train each of
        the trees.
    """
    def __init__(self, num_trees, max_tree_depth, ratio_per_tree=0.5):
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.ratio_per_tree=ratio_per_tree
        self.trees = []

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        if not isinstance(X, list):  # checking wheter X is a list or not, if not cast X to list
            X = X.tolist()
        if not isinstance(Y, list):  # checking wheter Y is a list or not, if not cast Y to list
            Y = Y.tolist()

        N_row = len(X)

        for num in range(self.num_trees):
            size = int(N_row * self.ratio_per_tree)# number of entries to use to train each tree
            indices = np.random.choice(N_row, size, replace=True) #randomly sample numbers from 0 to N, size of sample is size
            train_X = [X[i] for i in indices]
            train_Y = [Y[i] for i in indices]
            tree = DecisionTree(self.max_tree_depth)
            tree.fit(train_X, train_Y)
            self.trees.append(tree)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python
        list with labels, and conf being 1 dimensional list with
        confidences for each of the labels.
        """
        if not isinstance(X, list):  # checking wheter X is a list or not, if not cast X to list
            X = X.tolist()

        Y_trees = [tree.predict(X) for tree in self.trees] #2D list of predicted Y values for each tree, each row is a prediction list of a tree

        Y=[]
        conf=[]

        for single_pred in range(len(Y_trees[0])):
            values = [Y_trees[tree][single_pred] for tree in range(len(self.trees))] #selecting prediction values of all trees for the same row of X
            values = sorted(values)

            y_most = values[0] # most frequent y
            max_count = 1   #number of times y_most is observed
            cur_count = 1
            for i in range(1, len(values)):
                if values[i] != values[i - 1]: #we reached sequence of another value
                    if cur_count > max_count:
                        y_most = values[i - 1]
                        max_count = cur_count
                    cur_count = 1
                else:   #another observation of the same value
                    cur_count += 1

            if cur_count > max_count: #this means that last element of the list is the most frequent
                y_most = values[-1]
                max_count = cur_count

            Y.append(y_most)
            conf.append(max_count / self.num_trees)
        return (Y, conf)

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
        self.ratio_per_tree = ratio_per_tree
        self.trees = None

    def fit(self, X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array

        :return: forest - collection of decision trees
        """

        self.X = X
        self.Y = Y
        self.trees = []
        np.random.seed(13)
        n = len(X)
        sz = int(n * self.ratio_per_tree)

        if type(X) is np.ndarray:
            X = X.tolist()
        if type(Y) is np.ndarray:
            Y = Y.tolist()

        # randomly choosing a part of the data and fitting a corresponding tree for that data
        # repeat it num_tree times we get a random forest
        for i in range(self.num_trees):
            idx = np.arange(n)
            np.random.shuffle(idx)
            X = [X[idx[i]] for i in range(n)]
            Y = [Y[idx[i]] for i in range(n)]
            X_train = [X[i] for i in range(sz)]
            Y_train = [Y[i] for i in range(sz)]
            tree = DecisionTree(self.max_tree_depth)
            tree.fit(X_train, Y_train)
            self.trees.append(tree)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python
        list with labels, and conf being 1 dimensional list with
        confidences for each of the labels.

        """

        def column(matrix, i):
            return [row[i] for row in matrix]

        def most_common(lst):
            return max(set(lst), key=lst.count)

        Y = []
        predicted = []
        answer = []
        conf = []
        for i in range(self.num_trees):
            Y.append(self.trees[i].predict(X))
        for i in range(len(Y[0])):
            predicted.append(column(Y, i))
            answer.append(most_common(predicted[i]))
            conf.append(predicted[i].count(answer[i]) / len(predicted[i]))

        return (answer, conf)

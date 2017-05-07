from dtOmitted import DecisionNode
from dtOmitted import build_tree
import numpy as np
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
        """
        if not isinstance(X, list): #checking wheter X is a list or not, if not cast X to list
            X = X.tolist()
        if not isinstance(Y, list): #checking wheter Y is a list or not, if not cast Y to list
            Y = Y.tolist()

        data = [X[i][:] + [Y[i]] for i in range(len(X))] #appending Y to the last column of 2D X list
        self.root = build_tree(data, max_depth=self.max_depth)

    def single_predict(self,entry,node): # we will recursively deepen in tree until we reach a leaf
        if node.is_leaf==False:
            col_val=entry[node.column] # value of the entry at best_column
            best_val=node.value
            if type(best_val)==int or type(best_val)==float:
                if col_val>=best_val:
                    return self.single_predict(entry,node.true_branch)
                else:
                    return self.single_predict(entry,node.false_branch)
            else:
                if col_val==best_val:
                    return self.single_predict(entry, node.true_branch)
                else:
                    return self.single_predict(entry, node.false_branch)
        else:
            return node.result #if we reach a leaf return the result

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: Y - 1 dimension python list with labels
        """
        if not isinstance(X, list):  # checking wheter X is a list or not, if not cast X to list
            X = X.tolist()

        Y=[self.single_predict(entry,self.root) for entry in X]

        return Y

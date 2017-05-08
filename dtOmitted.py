import numpy as np
from collections import defaultdict

class DecisionNode(object):
    """
    README
    DecisionNode is a building block for Decision Trees.
    DecisionNode is a python class representing a  node in our decision tree
    node = DecisionNode()  is a simple usecase for the class
    you can also initialize the class like this:
    node = DecisionNode(column = 3, value = "Car")
    """

    def __init__(self,
                 column=None,
                 value=None,
                 false_branch=None,
                 true_branch=None,
                 current_results=None,
                 is_leaf=False,
                 result=None):
        """
        column is the column of the feature 
        value is the value of the feature of the specified column
        false_branch and true_branch are of type DecisionNode
        current_results is a dictionary that shows, for the current node,
            how many of each results it has (for example {"a":0, "b":5, "c":45})
        is_leaf is boolean and shows if node is a leaf
        result is the most popular answer from curren_results. (in the above example "c")
        """
        self.column = column
        self.value = value
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.current_results = current_results
        self.is_leaf = is_leaf
        max = -1
        for i in current_results.keys():
            a = current_results[i]
            if a > max:
                max = a
                self.result = i


def dict_of_values(data):
    """
    param data: a 2D Python list representing the data. Last column of data is Y.
    return: returns a python dictionary showing how many times each value appears in Y

    for example 
    data = [[1,'yes'],[1,'no'],[1,'yes'],[1,'yes']]
    dict_of_values(data)
    should return {'yes' : 3, 'no' :1}
    """
    results = defaultdict(int)
    for row in data:
        r = row[len(row) - 1]
        results[r] += 1
    return dict(results)


def divide_data(data, feature_column, feature_val):
    """
    this function dakes the data and divides it in two parts by a line. A line
    is defined by the feature we are considering (feature_column) and the target 
    value. The function returns a tuple (data1, data2) which are the desired parts of the data.
    For int or float types of the value, data1 have all the data with values >= feature_val
    in the corresponding column and data2 should have rest.
    For string types, data1 should have all data with values == feature val and data2 should 
    have the rest.

    param data: a 2D Python list representing the data. Last column of data is Y.
    param feature_column: an integer index of the feature/column.
    param feature_val: can be int, float, or string
    return: a tuple of two 2D python lists
    """
    data1 = []
    data2 = []
    if type(feature_val) == int or type(feature_val) == float:
        for i in range(len(data)):
            if data[i][feature_column] >= feature_val:
                data1.append(data[i])
            else:
                data2.append(data[i])
    if type(feature_val) == str:
        for i in range(len(data)):
            if data[i][feature_column] == feature_val:
                data1.append(data[i])
            else:
                data2.append(data[i])
    return (data1, data2)


def gini_impurity(data1, data2):
    """
    Given two 2D lists of compute their gini_impurity index. 
    Remember that last column of the data lists is the Y
    Lets assume y1 is y of data1 and y2 is y of data2.
    gini_impurity shows how diverse the values in y1 and y2 are.
    gini impurity is given by 

    N1*sum(p_k1 * (1-p_k1)) + N2*sum(p_k2 * (1-p_k2))

    where N1 is number of points in data1
    p_k1 is fraction of points that have y value of k in data1
    same for N2 and p_k2


    param data1: A 2D python list
    param data2: A 2D python list
    return: a number - gini_impurity 
    """
    dict1, dict2 = dict_of_values(data1), dict_of_values(data2)
    k1, k2 = list(dict1.values()), list(dict2.values())
    N1, N2 = sum(k1), sum(k2)

    p_k1, p_k2 = [x / N1 for x in k1], [x / N2 for x in k2]

    gini1 = 0
    gini2 = 0
    for i in range(len(k1)):
        gini1 += N1 * (p_k1[i] * (1 - p_k1[i]))
    for i in range(len(k2)):
        gini2 += N2 * (p_k2[i] * (1 - p_k2[i]))
    gini = gini1 + gini2

    return gini


def build_tree(data, current_depth=0, max_depth=1e10):
    """
    build_tree is a recursive function.
    What it does in the general case is:
    1: find the best feature and value of the feature to divide the data into
    two parts
    2: divide data into two parts with best feature, say data1 and data2
        recursively call build_tree on data1 and data2. this should give as two 
        trees say t1 and t2. Then the resulting tree should be 
        DecisionNode(...... true_branch=t1, false_branch=t2) 


    In case all the points in the data have same Y we should not split any more, and return that node.

    param data: param data: A 2D python list
    param current_depth: an integer. This is used if we want to limit the numbr of layers in the
        tree
    param max_depth: an integer - the maximal depth of the representing
    return: an object of class DecisionNode

    """
    if len(data) == 0:
        return DecisionNode(is_leaf=True)

    if (current_depth == max_depth):
        return DecisionNode(current_results=dict_of_values(data))

    if (len(dict_of_values(data)) == 1):
        return DecisionNode(current_results=dict_of_values(data), is_leaf=True)

    # This calculates gini number for the data before dividing
    self_gini = gini_impurity(data, [])

    # Below are the attributes of the best division that you need to find.
    # You need to update these when you find a division which is better
    best_gini = 1e10
    best_column = None
    best_value = None
    best_split = None

    def column(matrix, i):
        return [row[i] for row in matrix]

    for i in range(len(data[0]) - 1):
        for j in column(data, i):
            gini = gini_impurity(divide_data(data, i, j)[0], divide_data(data, i, j)[1])
            if gini < best_gini:
                best_gini = gini
                best_column = i
                best_value = j
                best_split = (divide_data(data, i, j)[0], divide_data(data, i, j)[1])
    # print(best_gini,best_column,best_value,best_split)

    # recursively call build tree, construct the correct return argument and return
    t1 = build_tree(best_split[1])
    t2 = build_tree(best_split[0])
    return DecisionNode(best_column, best_value, t1, t2, dict_of_values(data))


def print_tree(tree, indent=''):
    # Is this a leaf node?
    if tree.is_leaf:
        print(str(tree.current_results))
    else:
        # Print the criteria
        # print (indent+'Current Results: ' + str(tree.current_results))
        print('Column ' + str(tree.column) + ' : ' + str(tree.value) + '? ')

        # Print the branches
        print(indent + 'True->', end="")
        print_tree(tree.true_branch, indent + '  ')
        print(indent + 'False->', end="")
        print_tree(tree.false_branch, indent + '  ')

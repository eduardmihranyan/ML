import numpy as np

def normalize(X,l):
    X_normed=np.copy(X)
    X_var = np.var(X_normed, axis=0)   #normalizing regularization coefficient
    X_var[0] = 1
    l /= X_var
    l[0] = 0
    l = np.reshape(l, (l.shape[0], 1))
    X_bar = np.mean(X_normed, axis=0)
    X_std = np.std(X_normed, axis=0)
    for i in range(1, X_normed.shape[1]):
        X_normed[:, i] = (X_normed[:, i] - X_bar[i]) / X_std[i]

    return X_normed,l,X_bar,X_std

def weight_rescale(weights,mean,std):
    sumB = 0
    for i in range(1, weights.shape[0]):
        sumB += (mean[i] * weights[i]) / std[i]
        weights[i] /= std[i]
    weights[0] -= sumB

    return weights

def sigmoid(s):
    # You will find this function useful.
    return np.exp(s)/(1+np.exp(s))


def normalized_gradient(X, Y, beta, l):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: normalized gradient, i.e. gradient normalized according to data
    """
    beta=beta.reshape(beta.shape[0],1)
    grad=np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        sumL=0
        for j in range(X.shape[0]):
            sumL+=X[j,i]*Y[j]*(1-sigmoid(Y[j]*np.dot(beta.T, X[j, :])))
        grad[i]=l[i]*beta[i]-sumL
    grad/=X.shape[0]
    return grad


def gradient_descent(X, Y, epsilon=1e-6, l=1, step_size=1e-4, max_steps=1000):
    """
    Implement gradient descent using full value of the gradient.
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will
        terminate.
    :return: value of beta (1 dimensional np.array)
    """
    X_with_ones = np.column_stack((np.ones(len(X)), X)) #adding column of ones
    X_normed,l,X_bar,X_std=normalize(X_with_ones,l)
    beta = np.ones(X_with_ones.shape[1])
    for s in range(max_steps):
        grad = normalized_gradient(X_normed,Y, beta, l)
        if np.linalg.norm(step_size * grad) < epsilon:
            break
        beta -= step_size * grad

    weights=weight_rescale(beta,X_bar,X_std)
    return weights

def log_pred(X, weights):
    X_with_ones = np.column_stack(((np.ones(len(X))), X)) #adding a column of ones
    probs = [sigmoid(np.dot(x, weights)) for x in X_with_ones] #getting the probability of a data point to belong to class 1
    Y = [1 if prob > 0.5 else 0 for prob in probs] #classifying according to probabilities
    return Y

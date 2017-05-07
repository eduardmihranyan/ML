import numpy as np

def sigmoid(s):
    # TODO: Implement
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
    # TODO: Implement
    beta=beta.reshape(beta.shape[0],1)
    grad=np.ones(X.shape[1])
    for i in range(X.shape[1]):
        sum=0
        for j in range(X.shape[0]):
            sum+=X[j,i]*Y[j]*(1-sigmoid(Y[j]*beta.T.dot(X[j,:])))

        grad[i]=2*l*beta[i]-sum

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
    X_norm=np.copy(X)
    X_mean = np.mean(X,axis=0)
    X_std=np.std(X,axis=0)
    #print(X_std)
    for i in range(1,X.shape[1]):
        X_norm[:,i]=(X_norm[:,i]-X_mean[i])/X_std[i]
    X_var=np.var(X,axis=0)

    beta = np.ones(X.shape[1])
    for s in range(max_steps):
        #if s % 1000 == 0:
            #print(s, beta)
        # TODO: Implement iterations.
        grad_beta = normalized_gradient(X_norm, Y, beta, l)
        if np.linalg.norm(step_size * grad_beta) < epsilon:

            break
        else:
            beta = beta - step_size * grad_beta
        pass
    beta[0] = beta[0] - sum(X_mean[j] * beta[j] / X_std[j] for j in range(1, X.shape[1]))
    for i in range(1, X.shape[1]):
        beta[i] = beta[i] / X_std[i]
    return beta


def lr_predict(X, beta):
    """
    :param X: 2 dimensional python list or numpy 2 dimensional array representing the data to be predicted
    :param beta: 1 dimentional python list or numpy array representing the weights returned by our model 
        (in this case logistic regression)
    
    :return: 1 dimentional python list or numpy array, the predicted labels y_i for correspinding X_i
    """
    teta = [sigmoid(beta.T.dot(x)) for x in X]
    y = [1 if t > 0.5 else 0 for t in teta]
    return y
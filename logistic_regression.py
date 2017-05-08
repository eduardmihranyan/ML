import numpy as np


def sigmoid(s):
    return 1/(1+np.exp(-s))


def normalized_gradient(X, Y, beta, l):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: normalized gradient, i.e. gradient normalized according to data
    """

    # changing beta's shape,so that we can calculate it's transpose  e.g. (10,)-> (10,1)
    beta=beta.reshape(beta.shape[0],1)

    # calculating gradient
    grad=np.ones(X.shape[1])
    for i in range(X.shape[1]):
        sum=0
        for j in range(X.shape[0]):
            sum+=X[j,i]*Y[j]*(1-sigmoid(Y[j]*beta.T.dot(X[j,:])))

        grad[i]=2*l[i]*beta[i]-sum

    # dividing gradient by n=X.shape[0](gradient vector length) we get the normalized gradient
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

    # scaling X: (X-X_mean)/X_std

    for i in range(1,X.shape[1]):
        X_norm[:,i]=(X_norm[:,i]-X_mean[i])/X_std[i]
    X_var=np.var(X,axis=0)
    X_var[0]=1
    l=l/X_var
    l[0]=0


    #using gradient descent we get beta (notice that X has been scaled)
    beta = np.ones(X.shape[1])
    for s in range(max_steps):
        grad_beta = normalized_gradient(X_norm, Y, beta, l)
        if np.linalg.norm(step_size * grad_beta) < epsilon:

            break
        else:
            beta = beta - step_size * grad_beta
        pass

    # now we need to scale 'back' betas in order to get betas for the initial data
    beta[0] = beta[0] - sum(X_mean[j] * beta[j] / X_std[j] for j in range(1, X.shape[1]))
    for i in range(1, X.shape[1]):
        beta[i] = beta[i] / X_std[i]
    return beta


def lr_predict(X, beta):
    teta = [sigmoid(beta.T.dot(x)) for x in X]
    y = [1 if t > 0.5 else -1 for t in teta]
    return y
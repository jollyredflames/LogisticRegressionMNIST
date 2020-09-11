""" Methods for doing logistic regression."""
import numpy as np
from utils import sigmoid

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """
    data_with_bias_dimension = []

    for data_point in data:
        data_point = np.append(data_point, 1)
        data_with_bias_dimension.append(data_point)
    data_with_bias_dimension = np.array(data_with_bias_dimension)

    weights = np.array(weights)

    y = np.dot(data_with_bias_dimension, weights)

    return y

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    size = len(targets)
    value = [1 if i > 0.5 else 0 for i in y]
    correct = sum([1 if value[i] == targets [i] else 0 for i in range(len(targets))])

    y = sigmoid(y)

    ce = (-1 * targets * np.log(y) + -1 * (1 - targets) * np.log(1 - y))
    ce_sum = sum(ce)[0]

    frac_correct = correct / size
    ce = ce_sum / size

    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    N, M = data.shape

    z = logistic_predict(weights, data)

    #print(data.shape)
    new_data = np.insert(data, M, 1, axis=1)
    #print(new_data.shape)

    logOnePlusExpNeg = np.logaddexp(0, -1 * z)
    logOnePlusExpPos = np.logaddexp(0, z)
    # print(logPnePLusExpPos.shape, logPnePLusExpNeg.shape, N, M)

    f = sum(targets * logOnePlusExpNeg + (1 + -1 * targets) * logOnePlusExpPos)[0]
    # print(len(f))

    firstPart = -1 * np.exp(-1 * z) / (1 + np.exp(-1 * z))
    secondPart = np.exp(z) / (1 + np.exp(z))

    e_vector = targets * firstPart + (1 + -1 * targets) * secondPart

    df = np.matmul(np.transpose(new_data), e_vector)

    #print(f.shape)
    return f, df, y

def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:             N x 1 vector of probabilities.
    """

    f, df, y = logistic(weights, data, targets, hyperparameters)

    weightSquareSum = np.sum(np.square(weights))

    f = f + hyperparameters['weight_regularization'] / 2 * weightSquareSum
    df = df + hyperparameters['weight_regularization'] * np.sum(weights)

    return f, df, y

import numpy as np
from check_grad import check_grad
from utils import *
from logistic import *
import random
import matplotlib.pyplot as plt

def run_logistic_regression():
    train_inputs, train_targets = load_train()
    #train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_test()

    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.01,
                    'weight_regularization': 10,
                    'num_iterations': 400
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = []
    for _ in range(train_inputs.shape[1]+1):
        weights.append(random.uniform(0.01, 0.2))
    weights = np.array(weights)
    x = weights.shape[0]
    weights = weights.reshape((x, 1))

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    ceChartTrain = []
    ceChartValid = []
    errChartTrain = []
    errChartValid = []
    iteration = []
    # Begin learning with gradient descent
    for t in range(hyperparameters['num_iterations']):

        # TODO: you may need to modify this loop to create plots, etc.
        iteration.append(t)
        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)

        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)
        errChartTrain.append(1 - frac_correct_train)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        errChartValid.append(1 - frac_correct_valid)

        # print some stats
        print("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} \n TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}".format(
                   t+1, f / N, cross_entropy_train, frac_correct_train*100,
                   cross_entropy_valid, frac_correct_valid*100))

    plt.plot(iteration, errChartTrain, label="Train Acc")
    plt.plot(iteration, errChartValid, label="Valid Acc")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)

if __name__ == '__main__':
    run_logistic_regression()

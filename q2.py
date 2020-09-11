import matplotlib.pyplot as plt
from utils import *
from run_knn import *

if __name__ == "__main__":

    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    accuracy_for_k = []
    for i in [1, 3, 5, 7, 9]:
        data = run_knn(i, train_inputs, train_targets, test_inputs)
        accuracy_for_k.append(getAccuracy(data, test_targets))

    print(accuracy_for_k)

    plt.plot([1, 3, 5, 7, 9], accuracy_for_k, '-o')
    plt.title("K Value's Effect on Test Set")
    plt.xlabel("k value")
    plt.ylabel("Accuracy on Test Set")
    plt.show()


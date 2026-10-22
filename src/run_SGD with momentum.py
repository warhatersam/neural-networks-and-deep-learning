
import mnist_loader
import SDG_with_momentum as network
import random
import numpy as np
import matplotlib.pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 128, 128, 128, 10])

learning_rate = 0.1
batch_size = 128
steps = 1000
momentum = 0.9

def y_toVector(test_data):
    # test data (x, y) has y being a integer. We want y to be a 10-d vector,
    # so that it can be processed by net.calculate_avg_loss_and_gradient
    test_data_yVector = []
    for data in test_data:
        y_int = data[1]
        y_onehot = mnist_loader.vectorized_result(y_int)
        test_data_yVector.append((data[0], y_onehot))
    return test_data_yVector

def calculate_CE_loss(data):
    """
    take list of (x, y) where x, y are 10D vector
    do a forward pass through the network, calculate CE loss
    """
    sum = 0
    for x, y in data:
        z = net.feedforward(x)
        q = net.softmax(z)
        CE = np.sum(y * np.log(q))
        sum += CE
    loss = - (1/len(data))*sum
    return loss





if __name__ == "__main__":
    training_steps = []
    trainingLoss = []
    testLoss = []
    testdata_yOnehot = y_toVector(test_data)

    current_step = 0
    current_dataIndex = 0
    while current_step < steps:
        test_loss = calculate_CE_loss(testdata_yOnehot)
        loss = calculate_CE_loss(training_data)
        trainingLoss.append(loss)
        testLoss.append(test_loss)
        if current_dataIndex + batch_size >= len(training_data):
            net.update_parameters(training_data[current_dataIndex:], learning_rate, momentum)
            current_step += 1
            current_dataIndex = 0
            random.shuffle(training_data)
        else:
            batch_data = training_data[current_dataIndex: current_dataIndex + batch_size]
            net.update_parameters(batch_data, learning_rate, momentum)
            current_step += 1
            current_dataIndex += batch_size
        print(current_step)
        training_steps.append(current_step)




    plt.figure()
    plt.plot(training_steps, trainingLoss, label = "training loss")
    plt.plot(training_steps, testLoss, label = "test loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.title("SGD with momentum")
    plt.legend()
    plt.show()




import mnist_loader
import network4
import random
import numpy as np
import matplotlib.pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network4.Network([784, 30, 10])

batch_size = 10
learning_rate = 0.1

def training(epochs):
    for epoch in range(epochs):
        random.shuffle(training_data)
        loss_sum = 0
        for start in range(0, len(training_data), batch_size):
            batch = training_data[start:start + batch_size]
            loss = net.update_parameters(batch, learning_rate)
            loss_sum += loss
        avg_training_loss = loss_sum / len(training_data)
        print("Finished epoch", epoch + 1, "training loss:", avg_training_loss)
        return avg_training_loss

def calculate_accuracy_and_loss():
    correct = 0
    loss_sum = 0
    for x, y in test_data:
        final_z = net.feedforward(x)
        q = net.softmax(final_z)
        predicted_digit = np.argmax(q)
        if predicted_digit == y:
            correct += 1
        y_one_hot = mnist_loader.vectorized_result(y)
        loss = net.CE_for_single_datapoint(y_one_hot, final_z)[0]
        loss_sum += loss

    avg_loss = loss_sum / len(test_data)
    accuracy = correct / len(test_data)
    print("Test accuracy:", accuracy * 100, "%", "  Avg Loss:", avg_loss)
    return accuracy, avg_loss

if __name__ == "__main__":
    total_epoch = 30
    record_every = 500
    early_epoch_count = 2

    steps_per_epoch = (
                              len(training_data) + batch_size - 1
                      ) // batch_size
    early_step_limit = early_epoch_count * steps_per_epoch

    training_steps = []
    training_losses = []
    test_losses = []
    test_accuracy = []

    # Part 1(b): edit this list with test-image indices (not digit labels).
    tracked_indices = [3, 6, 8, 9]
    probability_steps = []
    probability_history = [[] for _ in tracked_indices]

    step = 0

    for epoch in range(total_epoch):
        random.shuffle(training_data)

        for start in range(0, len(training_data), batch_size):
            batch = training_data[start:start + batch_size]
            net.update_parameters(batch, learning_rate)
            step += 1

            final_step = (
                    epoch == total_epoch - 1
                    and start + batch_size >= len(training_data)
            )

            if step % record_every == 0 or final_step:
                # Training loss at the current, fixed parameters.
                loss_sum = 0.0

                for x, y in training_data:
                    final_z = net.feedforward(x)
                    loss = net.CE_for_single_datapoint(y, final_z)[0]
                    loss_sum += loss

                training_loss = loss_sum / len(training_data)

                # Test loss and accuracy.
                accuracy, test_loss = calculate_accuracy_and_loss()

                training_steps.append(step)
                training_losses.append(training_loss)
                test_losses.append(test_loss)
                test_accuracy.append(accuracy)

                # Part 1(b): track only the first 10 epochs.
                if step <= 10 * steps_per_epoch:
                    probability_steps.append(step)
                    for position, image_index in enumerate(tracked_indices):
                        x, y = test_data[image_index]
                        q = net.softmax(net.feedforward(x))
                        probability_history[position].append(q.flatten())

                print(
                    "Step:", step,
                    "Training loss:", training_loss,
                    "Test loss:", test_loss,
                    "Test accuracy:", accuracy
                )

        print("Finished epoch", epoch + 1)

    # Part 1(a): early measurements.
    early_count = sum(
        recorded_step <= early_step_limit
        for recorded_step in training_steps
    )

    early_steps = training_steps[:early_count]
    early_training = training_losses[:early_count]
    early_test = test_losses[:early_count]
    early_accuracy = test_accuracy[:early_count]

    # Early loss.
    plt.figure()
    plt.plot(early_steps, early_training, label="Training loss")
    plt.plot(early_steps, early_test, label="Test loss")
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Average cross-entropy")
    plt.title("Loss: first 10 epochs")
    plt.legend()
    plt.tight_layout()

    # Early accuracy.
    plt.figure()
    plt.plot(early_steps, early_accuracy)
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Test accuracy (fraction)")
    plt.title("Accuracy: first 10 epochs")
    plt.tight_layout()

    # Whole training history: loss.
    plt.figure()
    plt.plot(training_steps, training_losses, label="Training loss")
    plt.plot(training_steps, test_losses, label="Test loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Average cross-entropy")
    plt.title("Loss: all epochs")
    plt.legend()
    plt.tight_layout()

    # Whole training history: accuracy.
    plt.figure()
    plt.plot(training_steps, test_accuracy)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Test accuracy (fraction)")
    plt.title("Accuracy: all epochs")
    plt.tight_layout()

    # Part 1(b): one probability plot per selected image, up to epoch 10.
    for position, image_index in enumerate(tracked_indices):
        if not probability_steps:
            continue
        actual_digit = test_data[image_index][1]
        probabilities = np.array(probability_history[position])

        plt.figure()

        for digit in range(10):
            plt.plot(
                probability_steps,
                probabilities[:, digit],
                label=str(digit)
            )

        plt.xlabel("Training step")
        plt.ylabel("Predicted probability")
        plt.ylim(0, 1)
        plt.title(
            f"Image {image_index}, actual digit {actual_digit}: first 10 epochs"
        )
        plt.legend(title="Digit")
        plt.tight_layout()

    plt.show()

    # Whole training history: log-log accuracy
    plt.figure()
    plt.plot(training_steps, test_accuracy)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Test accuracy (fraction)")
    plt.title("Accuracy: all epochs")
    plt.tight_layout()

    plt.show()





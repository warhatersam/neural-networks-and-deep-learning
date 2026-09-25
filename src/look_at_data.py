import mnist_loader
import network4
import random
import numpy as np
import matplotlib.pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

index = 9
x, y = test_data[index]

plt.imshow(x.reshape(28, 28), cmap="gray")
plt.title(f"Test image {index}, actual digit: {y}")
plt.axis("off")
plt.show()

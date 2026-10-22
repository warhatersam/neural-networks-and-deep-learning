import numpy as np


class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        # a list of size for each layer
        self.weights = []
        self.biases = []
        self.m_weights = []
        self.m_biases = []
        self.v_weights = []
        self.v_biases = []


        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            W = np.random.randn(n_out, n_in) / np.sqrt(n_in)
            V_W = np.zeros_like(W)
            b = np.zeros((n_out, 1))
            V_b = np.zeros((n_out, 1))
            m_W = np.zeros_like(W)
            m_b = np.zeros_like(b)



            self.weights.append(W)
            self.biases.append(b)
            self.v_weights.append(V_W)
            self.v_biases.append(V_b)
            self.m_weights.append(m_W)
            self.m_biases.append(m_b)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    def softmax(self, z):
        shifted_z = z - np.max(z)
        exp_z = np.exp(shifted_z)
        return exp_z / np.sum(exp_z)
    def feedforward(self, x):
        # save intermediate z and activation functions
        self.z_for_each_laryer = []
        self.activation_for_each_layer = [x]

        for i in range(len(self.sizes) - 1):
            z = self.weights[i] @ self.activation_for_each_layer[i] + self.biases[i]
            activation = self.sigmoid(z)

            self.z_for_each_laryer.append(z)
            self.activation_for_each_layer.append(activation)

        return self.z_for_each_laryer[-1]

    def CE_for_single_datapoint(self, y, final_z):
        q = self.softmax(final_z)
        partial_loss_over_final_z = q - y
        return -np.sum(y * np.log(q)), partial_loss_over_final_z


    def calculate_avg_loss_and_gradient(self, data):
        loss_sum = 0
        sum_weight_grad = [np.zeros_like(w) for w in self.weights]
        sum_bias_grad = [np.zeros_like(b) for b in self.biases]
        for x, y in data:
            final_z = self.feedforward(x)
            loss, partial_loss_over_final_z = self.CE_for_single_datapoint(y, final_z)
            loss_sum += loss
            gradients_over_z = [None] * len(self.sizes)
            for l in range(len(self.weights), 0, -1):
                if l == len(self.weights):
                    gradients_over_z[l] = partial_loss_over_final_z
                    w_gradient = gradients_over_z[l] @ self.activation_for_each_layer[l-1].T
                    b_gradient = partial_loss_over_final_z
                    sum_weight_grad[l-1] += w_gradient
                    sum_bias_grad[l-1] += b_gradient
                else:
                    sigmoid_derivative = self.activation_for_each_layer[l] * (1 - self.activation_for_each_layer[l])
                    grad_over_zl = (self.weights[l].T @ gradients_over_z[l+1]) * sigmoid_derivative
                    gradients_over_z[l] = grad_over_zl
                    bl_gradient = grad_over_zl
                    sum_bias_grad[l-1] += bl_gradient
                    w_gradient = grad_over_zl @ self.activation_for_each_layer[l-1].T
                    sum_weight_grad[l-1] += w_gradient

        datasize = len(data)
        avg_loss = loss_sum / datasize
        avg_w_grad = [g / datasize for g in sum_weight_grad]
        avg_b_grad = [g / datasize for g in sum_bias_grad]

        return avg_loss, avg_w_grad, avg_b_grad

    def update_parameters(self, batch, learning_rate, b_1, b_2, lam, epsilon=1e-8):
        avg_loss, avg_w_grad, avg_b_grad = (
            self.calculate_avg_loss_and_gradient(batch)
        )
        for i in range(len(self.weights)):
            self.m_weights[i] = b_1 * self.m_weights[i] + (1 - b_1) * avg_w_grad[i]
            self.v_weights[i] = b_2 * self.v_weights[i] + (1 - b_2) * avg_w_grad[i] ** 2
            self.weights[i] = self.weights[i] * (1 - learning_rate * lam) - learning_rate * (self.m_weights[i] / (np.sqrt(self.v_weights[i]) + epsilon))
            self.m_biases[i] = b_1 * self.m_biases[i] + (1 - b_1) * avg_b_grad[i]
            self.v_biases[i] = b_2 * self.v_biases[i] + (1 - b_2) * avg_b_grad[i] ** 2
            self.biases[i] = self.biases[i] * (1 - learning_rate * lam) - learning_rate * (self.m_biases[i] / (np.sqrt(self.v_biases[i]) + epsilon))
        return avg_loss














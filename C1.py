import numpy as np

class SpatialNeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, dim):
        self.dim = dim
        self.input_layer = np.random.rand(input_neurons, dim)  # Initialize with specified dimensions
        self.hidden_layers = [np.random.rand(n, dim) for n in hidden_neurons]
        self.output_layer = np.random.rand(output_neurons, dim)

    def calculate_distances(self, layer1, layer2):
        diffs = layer1[np.newaxis, :, :] - layer2[:, np.newaxis, :]
        distances = np.sqrt(np.sum(diffs**2, axis=-1))
        return distances

    def feedforward(self, inputs):
        # Ensure input dimensions match
        if inputs.shape[-1] != self.dim:
            raise ValueError("Input dimensions must match the network's dimensionality")

        activations = inputs

        for i, hidden_layer in enumerate(self.hidden_layers):
            prev_layer = self.input_layer if i == 0 else self.hidden_layers[i-1]
            distances = self.calculate_distances(prev_layer, hidden_layer)
            activations = np.exp(-distances).dot(activations)

        output_distances = self.calculate_distances(self.hidden_layers[-1], self.output_layer)
        outputs = np.exp(-output_distances).dot(activations)

        return outputs

    def update_coordinates(self, layer, gradients, learning_rate):
        gradients_reshaped = gradients[:, np.newaxis]
        layer += learning_rate * gradients_reshaped

    def train(self, inputs, targets, learning_rate, epochs):
        if targets.shape[-1] != self.dim:
            raise ValueError("Target dimensions must match the network's output dimensionality")

        for epoch in range(epochs):
            outputs = self.feedforward(inputs)
            error = targets - outputs
            loss = np.mean(error**2)
            gradients = -2 * error.mean(axis=1)  # Average gradients over dimensions

            self.update_coordinates(self.output_layer, gradients, learning_rate)

            print(f'Epoch {epoch+1}, Loss: {loss}')

# Example usage
network = SpatialNeuralNetwork(input_neurons=3, hidden_neurons=[5, 5], output_neurons=10, dim=3)
inputs = np.random.rand(3, 3)  # Random inputs matching the dimension
targets = np.random.rand(10, 3)  # Random targets also in 3D

network.train(inputs, targets, learning_rate=0.01, epochs=100)
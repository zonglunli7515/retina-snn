import numpy as np
from reservoirpy.nodes import Reservoir, Ridge
import matplotlib.pyplot as plt

# Generate synthetic input and target data (e.g., a sine wave)
sequence_length = 100
input_data = np.sin(np.linspace(0, 10 * np.pi, sequence_length)).reshape(-1, 1)

# Create synthetic target data (e.g., 2D target for each time step)
target_data = np.column_stack([np.sin(np.linspace(0, 10 * np.pi, sequence_length)),
                               np.cos(np.linspace(0, 10 * np.pi, sequence_length))])

# Create a reservoir with specific parameters
reservoir = Reservoir(units=500, input_scaling=0.5, spectral_radius=0.9)

# Create a Ridge regression readout (output layer)
readout = Ridge(ridge=1e-6)

# Assemble the ESN
esn = reservoir >> readout

# Train the ESN with multidimensional target data
esn.fit(input_data, target_data)

# Test the ESN (here, we use the same input for simplicity)
predictions = esn.run(input_data)
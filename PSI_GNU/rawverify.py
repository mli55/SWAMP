import numpy as np

# Configure printing so you can see large arrays without ellipsis
np.set_printoptions(threshold=np.inf, linewidth=200)

filename = "/home/amelia/cfr_data.bin"
data = np.fromfile(filename, dtype=np.complex64)

print("Number of samples:", data.shape[0])

# Count how many samples are exactly zero.
# For complex data, `data == 0` checks if both real and imaginary parts are zero.
num_zeros = np.sum(data == 0)
fraction_zeros = num_zeros / data.shape[0] if data.shape[0] > 0 else 0

print(f"Number of zero-valued samples: {num_zeros}")
print(f"Fraction of zero-valued samples: {fraction_zeros:.4f}")

# You can add a simple threshold check if desired:
threshold = 0.1  # e.g., 10% threshold
if fraction_zeros > threshold:
    print(f"Warning: More than {threshold*100}% of the samples are zero! Possible desynchronization issue.")

# Print first few samples (instead of 5000) to get an overview:
print("First 10 samples:", data[:10])
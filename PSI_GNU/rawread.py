import numpy as np

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=200)

filename="/home/amelia/cfr_data.bin"
data=np.fromfile(filename, dtype=np.complex64)

print("numer of samples:", data.shape[0])
print("FIrst 10 samples:", data[:5000])

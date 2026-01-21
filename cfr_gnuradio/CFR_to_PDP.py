import os
import glob
import numpy as np
import pandas as pd
from scipy.fftpack import ifft
import ast

# Define the folder path containing the CFR data files
folder_path = '/home/amelia/CFR_nozero_raw/withoutPotato'

# Get a list of all TXT files in the folder
file_list = glob.glob(os.path.join(folder_path, '*.txt'))

def calculate_pdp(cfr_values):
    """
    Calculate Power Delay Profile (PDP) from CFR data.
    """
    # Perform Inverse FFT to convert CFR to time domain
    time_domain_signal = ifft(cfr_values)
    
    # Calculate power delay profile by taking the squared magnitude
    pdp = np.abs(time_domain_signal) ** 2
    return pdp

for file_path in file_list:
    # Read the CFR data from the .txt file
    with open(file_path, 'r') as f:
        file_contents = f.read().strip()
        
    # Parse the list of complex numbers from the file
    # The file is expected to contain data in a Python-list-like format: [(...),(...), ...]
    cfr_values = ast.literal_eval(file_contents)  # returns a list of complex numbers

    # Convert to numpy array if not already
    cfr_values = np.array(cfr_values, dtype=complex)

    # Calculate PDP
    pdp = calculate_pdp(cfr_values)

    # Create a DataFrame to hold PDP results
    # Since we don't have distance information, we'll just store indices
    pdp_df = pd.DataFrame({
        'TimeIndex': np.arange(len(pdp)),
        'PDP': pdp
    })

    # Construct the output file name
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(folder_path, f"{base_name}_PDP.csv")

    # Save the PDP data to a new CSV file
    pdp_df.to_csv(output_file, index=False)
    print(f"Processed {file_path}, results saved to {output_file}")

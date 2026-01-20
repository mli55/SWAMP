import os
import statistics

# Specify the root folder
root_folder = '/Users/tengfei/Desktop/PSI_100points_other_factors'

# Specify the output file (in the root folder)
output_file = os.path.join(root_folder, 'stats.txt')

with open(output_file, 'w') as summary:
    # Write a header for the output file (optional)
    summary.write("File\tMean_RSRP\tVariance_RSRP\tMean_MCS\tVariance_MCS\n")
    
    # Walk through all subdirectories and files
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.path.join(dirpath, filename)

                # Read all lines from the file
                with open(file_path, 'r') as infile:
                    lines = infile.readlines()

                # Extract RSRP and MCS values
                rsrp_values = []
                mcs_values = []

                for line in lines:
                    # Split the line by spaces
                    parts = line.split()
                    
                    # Ensure we have at least 2 columns (RSRP and MCS)
                    if len(parts) < 2:
                        continue
                    
                    # Try converting RSRP and MCS to floats
                    try:
                        rsrp = float(parts[0])
                        mcs = float(parts[1])
                        rsrp_values.append(rsrp)
                        mcs_values.append(mcs)
                    except ValueError:
                        # If conversion fails, skip this line
                        continue

                # Compute mean and variance if we have data
                if rsrp_values and mcs_values:
                    mean_rsrp = statistics.mean(rsrp_values)
                    var_rsrp = statistics.variance(rsrp_values) if len(rsrp_values) > 1 else 0.0
                    mean_mcs = statistics.mean(mcs_values)
                    var_mcs = statistics.variance(mcs_values) if len(mcs_values) > 1 else 0.0
                    
                    # Write result to the summary file
                    summary.write(f"{file_path}\t{mean_rsrp}\t{var_rsrp}\t{mean_mcs}\t{var_mcs}\n")
                else:
                    # If no valid data lines, write NoData
                    summary.write(f"{file_path}\tNoData\tNoData\tNoData\tNoData\n")

print(f"Summary stats saved to {output_file}")

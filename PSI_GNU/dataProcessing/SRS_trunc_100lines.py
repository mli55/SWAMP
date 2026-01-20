import os

# Specify the root folder containing subfolders
root_folder = '/Users/tengfei/Desktop/PSI_100points_other_factors'

for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.endswith('.txt'):
            input_file = os.path.join(dirpath, filename)
            
            # Read the file lines first
            with open(input_file, 'r') as infile:
                lines = infile.readlines()
            
            # Check if the first two lines match the unwanted lines
            if (len(lines) > 2 and 
                lines[0].strip() == "RSRP (dB) | MCS | SINR (dB) | DL Brate (bps) | BLER (%)" and
                lines[1].strip().startswith("----------")):
                
                # Skip the first two lines
                truncated_lines = lines[2:2+100]  # Take 100 lines starting from line 3
            else:
                # No special header, just take the first 100 lines
                truncated_lines = lines[:100]
            
            # Overwrite the original file with the truncated content
            with open(input_file, 'w') as outfile:
                outfile.writelines(truncated_lines)
            
            print(f"Truncated {input_file} to the first 100 lines after removing header if present.")

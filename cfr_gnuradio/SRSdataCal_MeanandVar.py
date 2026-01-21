import pandas as pd

# Load the CSV file
file_path = '/home/tengfei/PSI_GNU/Data_Collection_FixedShort_withP/withPotatoShort.csv'
data = pd.read_csv(file_path)

# Separate the last column as the indicator and the rest as numeric columns
numeric_columns = data.iloc[:, :-1]
indicator_column = data.iloc[:, -1]

# Compute the mean and variance grouped by the indicator column
grouped_stats = numeric_columns.groupby(indicator_column).agg(['mean', 'var'])

# Flatten the multi-index columns for better readability
grouped_stats.columns = ['_'.join(col).strip() for col in grouped_stats.columns.values]

# Save the results to a new CSV file
output_file_path = '/home/tengfei/PSI_GNU/Data_Collection_FixedShort_withP/without_PShort_MeanVar.csv'
grouped_stats.to_csv(output_file_path)

print(f"Results have been saved to: {output_file_path}")

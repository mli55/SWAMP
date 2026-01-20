import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/home/tengfei/PSI_GNU/PSI_SRSdata - meanandVar.csv'
data = pd.read_csv(file_path)

# Extract the columns to be plotted
x_column = 'DISTANCE'
condition_column = 'Condition'
y_columns = [col for col in data.columns if col not in [x_column, condition_column]]

# Ensure y_columns contain numeric data
data[y_columns] = data[y_columns].apply(pd.to_numeric, errors='coerce')

# Get unique conditions
conditions = data[condition_column].unique()

# Create a figure for each y_column
for y_col in y_columns:
    plt.figure(figsize=(10, 6))
    
    # Plot data for each condition
    for condition in conditions:
        subset = data[data[condition_column] == condition]
        plt.plot(subset[x_column].to_numpy(), subset[y_col].to_numpy(), label=f"{condition}")
    
    # Customize the plot
    plt.title(f"{y_col} vs {x_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_col)
    plt.legend(title=condition_column)
    plt.grid(True)
    
    # Display the plot
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the 2019 Maha season dataset
file_path = r"E:/My_Research/Crop/Codes/output/2019_Maha_final_with_yield.csv"  # Adjust the file path
df = pd.read_csv(file_path)

# Print column names to check for any inconsistencies
print("Columns in dataset:", df.columns)

# Clean column names (strip leading/trailing spaces and convert to lowercase)
df.columns = df.columns.str.strip().str.lower()

# Define the target variable and predictor columns (use lowercase after cleaning)
y_variable = 'landsat_8'
predictor_columns = ['temp', 'rain', 'evapo', 'daylight', 'evi', 'ndmi', 'ndvi', 'savi',
                     'ec', 'k', 'p', 'ph', 'zn']

# Subset the data to include only relevant columns and drop rows with missing values
valid_columns = [col for col in predictor_columns if col in df.columns]
df_subset = df[[y_variable] + valid_columns].dropna()

# Calculate Pearson correlation for all variables
correlation_matrix = df_subset.corr(method='pearson')

# Save the full correlation matrix to a CSV file
output_matrix_file = r"E:/My_Research/Crop/Data/outputs/2019_Maha_correlation_matrix.csv"
correlation_matrix.to_csv(output_matrix_file)
print(f"Full correlation matrix saved to: {output_matrix_file}")

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Pearson Correlation Matrix (2019 Maha Season)")
plt.tight_layout()

# Save the heatmap
output_heatmap_file = r"E:/My_Research/Crop/Data/outputs/2019_Maha_correlation_heatmap.png"
plt.savefig(output_heatmap_file)
plt.show()

print(f"Heatmap saved to: {output_heatmap_file}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path = "Pond Readings.xlsx"  # Update with your file path
xls = pd.ExcelFile(file_path)

# Load the data from the sheet
df = xls.parse("Pond Readings")

# Select only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['number'])

# Compute the correlation matrix
correlation_matrix = numeric_df.corr()

# Extract correlation with "Net OD"
correlation_with_net_od = correlation_matrix["Net OD"].drop("Net OD")

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_net_od.index, y=correlation_with_net_od.values, palette="viridis")

# Customize the plot
plt.xticks(rotation=45, ha='right')
plt.xlabel("Parameters")
plt.ylabel("Correlation with Net OD")
plt.title("Correlation of Parameters with Net OD")
plt.grid(axis="y")

# Show the plot
plt.show()

# Print correlation values
print(correlation_with_net_od)

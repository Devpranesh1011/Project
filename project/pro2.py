import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr

# Step 1: Load Data from Excel
file_path = "Pond Readings.xlsx"  # Replace with the path to your Excel file
df = pd.read_excel(file_path, parse_dates=["Date"], index_col="Date")  # Ensure 'Date' column is present

# Preview the data
print("Data Snapshot:")
print(df.head())

# Check for missing data
if df.isnull().sum().any():
    print("\nMissing Data Summary:")
    print(df.isnull().sum())
    print("Filling missing values using forward-fill...")
    df.fillna(method="ffill", inplace=True)

# Step 2: Exploratory Data Analysis (EDA)
print("\nDescriptive Statistics:")
print(df.describe())

# Plot trends for each parameter
for column in df.columns:
    if column != "Pond":  # Exclude categorical columns like "Pond" from trends
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x=df.index, y=column, hue="Pond" if "Pond" in df.columns else None)
        plt.title(f"Trend of {column} Over Time")
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.legend(loc="upper right")
        plt.grid()
        plt.show()

# Step 3: Correlation Analysis
# Compute and visualize correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Display high correlation pairs
threshold = 0.7  # Adjust this threshold as needed
print("\nHigh Correlation Pairs:")
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            print(f"{correlation_matrix.columns[i]} and {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]:.2f}")

# Step 4: Temporal Relationships (Lag Analysis)
# Example: Cross-correlation for "Ambient Illuminance (lx)" and "Water Temperature (°C)"
def plot_cross_correlation(series1, series2, lag_range=30):
    lags = range(-lag_range, lag_range + 1)
    correlations = [series1.corr(series2.shift(lag)) for lag in lags]
    plt.figure(figsize=(8, 4))
    plt.plot(lags, correlations, marker="o")
    plt.axhline(0, color="gray", linestyle="--")
    plt.title(f"Cross-Correlation Plot ({series1.name} vs {series2.name})")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.grid()
    plt.show()

if "Ambient Illuminance (lx)" in df.columns and "Water Temperature (°C)" in df.columns:
    plot_cross_correlation(df["Ambient Illuminance (lx)"], df["Water Temperature (°C)"])

# Granger Causality Test Example: Check if "Salinity (‰)" can predict "Water Temperature (°C)"
if "Salinity (‰)" in df.columns and "Water Temperature (°C)" in df.columns:
    print("\nGranger Causality Test Results:")
    grangercausalitytests(df[["Water Temperature (°C)", "Salinity (‰)"]].dropna(), maxlag=5, verbose=True)

# Step 5: Summary of Findings
print("\nSummary of Findings:")
print("- Correlation analysis shows relationships between various parameters.")
print("- Temporal analysis identifies potential lag effects between parameters.")

# Example: Scatter plot to visualize two correlated parameters
if "Hardness" in df.columns and "pH" in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Hardness", y="pH", hue="Pond" if "Pond" in df.columns else None)
    plt.title("Scatter Plot of Hardness vs pH")
    plt.xlabel("Hardness")
    plt.ylabel("pH")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()

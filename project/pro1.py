import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from statsmodels.tsa.stattools import grangercausalitytests # type: ignore
from scipy.stats import pearsonr # type: ignore
file_path = "Pond_Readings.csv"  
df = pd.read_excel(file_path, parse_dates=["Date"], index_col="Date")
print("Data Snapshot:")
print(df.head())
if df.isnull().sum().any():
    print("\nMissing Data Summary:")
    print(df.isnull().sum())
    print("Filling missing values using forward-fill...")
    df.fillna(method="ffill", inplace=True)
print("\nDescriptive Statistics:")
print(df.describe())
for column in df.columns:
    if column != "Pond":  
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x=df.index, y=column, hue="Pond" if "Pond" in df.columns else None)
        plt.title(f"Trend of {column} Over Time")
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.legend(loc="upper right")
        plt.grid()
        plt.show()
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
threshold = 0.7  
print("\nHigh Correlation Pairs:")
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            print(f"{correlation_matrix.columns[i]} and {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]:.2f}")
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
if "Salinity (‰)" in df.columns and "Water Temperature (°C)" in df.columns:
    print("\nGranger Causality Test Results:")
    grangercausalitytests(df[["Water Temperature (°C)", "Salinity (‰)"]].dropna(), maxlag=5, verbose=True)
print("\nSummary of Findings:")
print("- Correlation analysis shows relationships between various parameters.")
print("- Temporal analysis identifies potential lag effects between parameters.")
if "Hardness" in df.columns and "pH" in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Hardness", y="pH", hue="Pond" if "Pond" in df.columns else None)
    plt.title("Scatter Plot of Hardness vs pH")
    plt.xlabel("Hardness")
    plt.ylabel("pH")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()

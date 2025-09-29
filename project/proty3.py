import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Load data
file_path = r"C:\Users\user\Documents\pranish\OneDrive\Desktop\Pond Readings.xlsx"
df = pd.read_excel(file_path)

# Convert 'Date' column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Drop rows with missing values
df = df.dropna()

# Calculate correlation matrix
corr_matrix = df.select_dtypes(include=['number']).corr()

# Plot correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Pond Parameters")
plt.show()

# Group data by 'Date' and calculate mean
df_daily = df.groupby("Date").mean(numeric_only=True)

# Plot daily parameters
fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

axes[0].plot(df_daily.index, df_daily["Water Temperature (°C)"], marker='o', linestyle='-', color='b', label="Water Temp (°C)")
axes[0].set_ylabel("Temperature (°C)")
axes[0].set_title("Water Temperature Over Time")
axes[0].legend()

axes[1].plot(df_daily.index, df_daily["pH"], marker='s', linestyle='-', color='g', label="pH Level")
axes[1].set_ylabel("pH Level")
axes[1].set_title("pH Levels Over Time")
axes[1].legend()

axes[2].plot(df_daily.index, df_daily["Salinity (‰)"], marker='d', linestyle='-', color='r', label="Salinity (‰)")
axes[2].set_ylabel("Salinity (‰)")
axes[2].set_title("Salinity Over Time")
axes[2].legend()

axes[3].plot(df_daily.index, df_daily["Net OD"], marker='x', linestyle='-', color='m', label="Net OD")
axes[3].set_ylabel("Net OD")
axes[3].set_title("Net OD Over Time")
axes[3].legend()

plt.xlabel("Date")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Select features and target
features = ["pH", "Salinity (‰)", "Ambient Illuminance (lx)", "Pond Depth (cm)", "Net OD"]
target = "Water Temperature (°C)"

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Create and train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
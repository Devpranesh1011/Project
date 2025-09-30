# Project
# CODE: 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error, r2_score 
file_path = r"C:\Users\user\Documents\pranesh\OneDrive\Desktop\Pond Readings.xlsx" 
df = pd.read_excel(file_path) 
df.columns = df.columns.str.strip() 
df = df.rename(columns={ 
"Salinity (â€°)": "Salinity", 
"Water Temperature (Â°C)": "Water Temperature (°C)" 
}) 
df["Date"] = pd.to_datetime(df["Date"]) 
df = df.dropna() 
df = df.sort_values("Date") 
lag_features = ["pH", "Salinity", "Ambient Illuminance (lx)",  
"Pond Depth (cm)", "Net OD", "Water Temperature (°C)"] 
for feature in lag_features: 
if feature in df.columns: 
16 
df[f"{feature}_lag1"] = df[feature].shift(1) 
else: 
print(f"Warning: {feature} not found in dataset") 
df = df.dropna() 
corr_matrix = df.select_dtypes(include=['number']).corr() 
plt.figure(figsize=(10,6)) 
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5) 
plt.title("Correlation Matrix of Pond Parameters") 
plt.show() 
df_daily = df.groupby("Date")[df.select_dtypes(include=["number"]).columns].mean() 
fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True) 
axes[0].plot(df_daily.index, df_daily["Water Temperature (°C)"], marker='o', linestyle='-', color='b', 
label="Water Temp (°C)") 
axes[0].set_ylabel("Temperature (°C)") 
axes[0].set_title("Water Temperature Over Time") 
axes[0].legend() 
axes[1].plot(df_daily.index, df_daily["pH"], marker='s', linestyle='-', color='g', label="pH Level") 
axes[1].set_ylabel("pH Level") 
axes[1].set_title("pH Levels Over Time") 
axes[1].legend() 
axes[2].plot(df_daily.index, df_daily["Salinity"], marker='d', linestyle='-', color='r', label="Salinity") 
axes[2].set_ylabel("Salinity") 
axes[2].set_title("Salinity Over Time") 
axes[2].legend() 
axes[3].plot(df_daily.index, df_daily["Net OD"], marker='x', linestyle='-', color='m', label="Net OD") 
17 
axes[3].set_ylabel("Net OD") 
axes[3].set_title("Net OD Over Time") 
axes[3].legend() 
plt.xlabel("Date") 
plt.xticks(rotation=45) 
plt.tight_layout() 
plt.show() 
features = [f"{feature}_lag1" for feature in lag_features if feature != "Water Temperature (°C)"] 
target = "Water Temperature (°C)" 
missing_features = [f for f in features if f not in df.columns] 
if missing_features: 
print(f"Warning: Missing Features -> {missing_features}") 
features = [f for f in features if f in df.columns] 
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42) 
rf_model = RandomForestRegressor(n_estimators=100, random_state=42) 
rf_model.fit(X_train, y_train) 
y_pred = rf_model.predict(X_test) 
mae = mean_absolute_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 
print(f"Mean Absolute Error: {mae:.2f}") 
print(f"R² Score: {r2:.2f}") 
feature_importance = pd.Series(rf_model.feature_importances_, 
index=features).sort_values(ascending=False) 
plt.figure(figsize=(8,6)) 
18 
sns.barplot(x=feature_importance, y=feature_importance.index, palette="viridis") 
plt.xlabel("Feature Importance Score") 
plt.ylabel("Features") 
plt.title("Feature Importance in Random Forest Model") 
plt.show() 
plt.figure(figsize=(8,6)) 
sns.scatterplot(x=y_test.values, y=y_pred, alpha=0.7) 
plt.xlabel("Actual Water Temperature (°C)") 
plt.ylabel("Predicted Water Temperature (°C)") 
plt.title("Actual vs Predicted Water Temperature") 
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

print("=== Task 4: Network Traffic Prediction (Min MSE, Max R²) ===")

# Load SDN traffic data
df = pd.read_csv("/home/ubuntu/Downloads/SDN_traffic.csv")
print(f"Dataset: {df.shape[0]} flows")

# Target: Predict forward_size_bytes (traffic volume)
target_col = 'forward_size_bytes'
y = df[target_col]
print(f"Predicting: {target_col}")
print(f"Target range: {y.min():.0f} - {y.max():.0f} bytes")

# Features (drop ID, category, target)
X = df.drop(['id_flow', 'category', target_col], axis=1, errors='ignore')

# Create time-based features for prediction (week simulation)
df['duration_hours'] = df['forward_duration'] / 3600  # seconds to hours
X['duration_hours'] = df['duration_hours']
X['pps'] = df['forward_pps']  # packets per second
X['bps'] = df['forward_bps']  # bytes per second

# Encode categoricals
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Lag features (previous traffic predicts future)
X['bytes_lag1'] = y.shift(1).fillna(y.mean())
X['bytes_lag2'] = y.shift(2).fillna(y.mean())
X = X.dropna()

y = y[2:]  # align with lags
X = X.iloc[2:]

# Train-test split (time series: no shuffle)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# RandomForestRegressor (minimize MSE)
rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_train)

# Predictions
y_pred_train = rf_reg.predict(X_train)
y_pred_test = rf_reg.predict(X_test)

# Metrics (YOUR GOALS)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("\n" + "="*60)
print("TRAFFIC PREDICTION RESULTS")
print(f"TRAIN MSE:  {train_mse:,.0f}")
print(f"TEST  MSE: {test_mse:,.0f} (MINIMIZED!)")
print(f"TRAIN R²:  {train_r2:.4f}")
print(f"TEST  R²:  {test_r2:.4f} (MAXIMIZED!)")

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_reg.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(importances.head(10).round(4))

# Plots for report
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Actual vs Predicted
axes[0,0].scatter(y_test, y_pred_test, alpha=0.6)
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0,0].set_xlabel('Actual Bytes')
axes[0,0].set_ylabel('Predicted Bytes')
axes[0,0].set_title(f'Actual vs Predicted\nR² = {test_r2:.4f}')

# Residuals
residuals = y_test - y_pred_test
axes[0,1].scatter(y_pred_test, residuals, alpha=0.6)
axes[0,1].axhline(0, color='r', linestyle='--')
axes[0,1].set_xlabel('Predicted')
axes[0,1].set_ylabel('Residuals')
axes[0,1].set_title(f'Residuals (MSE = {test_mse:,.0f})')

# Feature Importance
top10 = importances.head(10)
sns.barplot(data=top10, x='importance', y='feature', ax=axes[1,0])
axes[1,0].set_title('Top 10 Feature Importance')

# Time series prediction
test_idx = range(len(y_test))
axes[1,1].plot(test_idx, y_test.values, label='Actual', alpha=0.7)
axes[1,1].plot(test_idx, y_pred_test, label='Predicted', alpha=0.7)
axes[1,1].set_xlabel('Test Samples')
axes[1,1].set_ylabel('Bytes')
axes[1,1].legend()
axes[1,1].set_title('Traffic Prediction Over Time')

plt.tight_layout()
plt.savefig('task4_traffic_prediction.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results table
results = pd.DataFrame({
    'Metric': ['Train MSE', 'Test MSE', 'Train R²', 'Test R²'],
    'Value': [train_mse, test_mse, train_r2, test_r2]
})
results.to_csv('task4_results.csv', index=False)
importances.to_csv('task4_feature_importance.csv', index=False)

print("\n✅ SAVED:")
print("  task4_traffic_prediction.png (4 plots)")
print("  task4_results.csv")
print("  task4_feature_importance.csv")
print("\n=== TASK 4 COMPLETE! 100% ASSIGNMENT DONE! 🎉 ===")

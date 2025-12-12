import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

print("=== Task 2: SDN Traffic Classification (ID3 vs CART) ===")

# Load YOUR SDN CSV
df = pd.read_csv("/home/ubuntu/Downloads/SDN_traffic.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 3 rows:")
print(df.head(3))
print("\nColumn names:")
print(df.columns.tolist())

# Use 'category' column (detected from your data)
target_col = 'category'
print(f"\nUsing target column: '{target_col}'")
print(f"Label distribution:\n{df[target_col].value_counts()}")

# Preprocessing
y = df[target_col]
X = df.drop(columns=[target_col])

# Encode categorical features (id_flow, nw_src, nw_dst)
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target
y_le = LabelEncoder()
y = y_le.fit_transform(y)
print(f"Encoded classes: {dict(zip(y_le.classes_, range(len(y_le.classes_))))}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ID3-like (Entropy)
print("\n" + "="*60)
print("1. ID3-like Decision Tree (criterion='entropy')")
dt_id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_id3.fit(X_train, y_train)
y_pred_id3 = dt_id3.predict(X_test)
acc_id3 = accuracy_score(y_test, y_pred_id3)
f1_id3 = f1_score(y_test, y_pred_id3, average='macro')
print(f"Accuracy: {acc_id3:.4f}")
print(f"F1-score (macro): {f1_id3:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_id3, target_names=y_le.classes_))

# CART (Gini)
print("\n" + "="*60)
print("2. CART Decision Tree (criterion='gini')")
dt_cart = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_cart.fit(X_train, y_train)
y_pred_cart = dt_cart.predict(X_test)
acc_cart = accuracy_score(y_test, y_pred_cart)
f1_cart = f1_score(y_test, y_pred_cart, average='macro')
print(f"Accuracy: {acc_cart:.4f}")
print(f"F1-score (macro): {f1_cart:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_cart, target_names=y_le.classes_))

# Comparison table
comparison = pd.DataFrame({
    'Algorithm': ['ID3-like (Entropy)', 'CART (Gini)'],
    'Accuracy': [acc_id3, acc_cart],
    'F1_macro': [f1_id3, f1_cart]
})
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print(comparison.round(4))

# Save comparison table to CSV for report
comparison.to_csv('task2_comparison_table.csv', index=False)
print("✅ SAVED: task2_comparison_table.csv")

# Confusion matrices plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_id3), annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title('ID3-like (Entropy) Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test, y_pred_cart), annot=True, fmt='d', ax=axes[1], cmap='Blues')
axes[1].set_title('CART (Gini) Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('task2_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ SAVED: task2_comparison.png")
print("\n=== Task 2 COMPLETE! ===")
print("For report: Copy accuracy table + screenshot PNG + CSV file")
print("\nGit commands:")
print("git add task2_sdn_dt.py task2_comparison.png task2_comparison_table.csv")
print("git commit -m 'Task 2: SDN Traffic Classification ID3 vs CART'")
print("git push")

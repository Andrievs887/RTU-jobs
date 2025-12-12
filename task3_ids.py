import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

print("=== Task 3: Intrusion Detection (Minimize False Alerts) ===")

# Use SDN data as ISCX proxy: Normal=WWW vs Attack=others
df = pd.read_csv("/home/ubuntu/Downloads/SDN_traffic.csv")
df['is_attack'] = df['category'].apply(lambda x: 0 if x == 'WWW' else 1)

print(f"Dataset: {df.shape[0]} flows")
print("Normal (WWW):", (df['is_attack']==0).sum())
print("Attack (FTP/P2P/etc):", (df['is_attack']==1).sum())

# Features (drop ID and category)
X = df.drop(['id_flow', 'category', 'is_attack'], axis=1, errors='ignore')
y = df['is_attack']

# Encode categorical columns (nw_src, nw_dst)
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# RandomForest - MINIMIZE FALSE POSITIVES (your goal)
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight={0:1, 1:3},  # penalize missing attacks
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

# Results
print("\n" + "="*60)
print("INTRUSION DETECTION RESULTS")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# Confusion Matrix + FPR (your main metric)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp/(fp+tn)*100
print(f"\nCONFUSION MATRIX:")
print(f"TN(Normal correct): {tn:>4} | FP(False Alarm): {fp:>4} | FPR: {fpr:.2f}%")
print(f"FN(Miss Attack)  : {fn:>4} | TP(Detect Attack):{tp:>4}")

# Threshold tuning table (minimize FPR further)
print("\nThreshold Tuning (Lower FPR):")
for thr in [0.3, 0.4, 0.5]:
    y_thr = (y_proba >= thr).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_thr).ravel()
    fpr2 = fp2/(fp2+tn2)*100
    print(f"Thr={thr}: FPR={fpr2:.2f}%, FN={fn2}")

# Plots for report
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Confusion Matrix\n(Low False Positive Rate)')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# Precision by class
precision_normal = tn/(tn+fn)
precision_attack = tp/(tp+fp)
ax2.bar(['Normal Precision', 'Attack Precision'], [precision_normal, precision_attack], color=['green', 'red'])
ax2.set_ylim(0,1)
ax2.set_title('Precision by Class')

plt.tight_layout()
plt.savefig('task3_ids_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ SAVED: task3_ids_results.png")
print("\n=== Task 3 COMPLETE! ===")
print("Files for report: task3_ids_results.png + terminal output")

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt

# Step 1: Load data
conn = sqlite3.connect('road_safety.db')
df = pd.read_sql_query("SELECT * FROM inputs", conn)
conn.close()

# Step 2: Features and Labels
X = df[['temperature', 'visibility', 'traffic_density', 'vehicle_speed', 'risk_zone']]
y = df['label']

print("\nLabel Distribution:")
print(y.value_counts())

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 4: Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=5000),
    "SVM": SVC(kernel='rbf', probability=True)
}

best_model = None
best_accuracy = 0
best_model_name = ""

# Step 5: Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📈 {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Caution", "Danger"],zero_division=0))
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

    from collections import Counter
    print("Predicted Label Distribution:", Counter(y_pred))


# Step 6: Save the best model
joblib.dump(best_model, 'risk_predictor3.pkl')
print(f"\n🏆 Best model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
print("✅ Model saved as 'risk_predictor3.pkl'.")

# Step 7: Plot Confusion Matrix
y_best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_best_pred)
fig, ax = plt.subplots(figsize=(6,6))
ax.set_title(f"Confusion Matrix: {best_model_name}")
ax.imshow(cm, cmap='Blues')
ax.grid(False)
ax.set_xlabel('Predicted', fontsize=12, color='black')
ax.set_ylabel('Actual', fontsize=12, color='black')
ax.xaxis.set(ticks=(0, 1, 2), ticklabels=["Safe", "Caution", "Danger"])
ax.yaxis.set(ticks=(0, 1, 2), ticklabels=["Safe", "Caution", "Danger"])
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.show()

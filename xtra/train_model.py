import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- 1. Load Data -------------------
conn = sqlite3.connect("road_safety.db")
df = pd.read_sql_query("SELECT temperature, visibility, traffic_density, vehicle_speed, risk_zone FROM inputs", conn)
conn.close()

# Add small random noise
df['vehicle_speed'] += np.random.normal(0, 5, df.shape[0])
df['visibility'] += np.random.normal(0, 100, df.shape[0])
df['traffic_density'] += np.random.normal(0, 5, df.shape[0])

# Clip to valid ranges
df['vehicle_speed'] = df['vehicle_speed'].clip(0, 130)
df['visibility'] = df['visibility'].clip(0, 10000)
df['traffic_density'] = df['traffic_density'].clip(0, 100)

# ------------------- 2. Label Risk -------------------
def label_risk(row):
    score = 0
    if row['vehicle_speed'] > 90:
        score += 6
    elif row['vehicle_speed'] > 70:
        score += 3
    elif row['vehicle_speed'] > 50:
        score += 1
    if row['visibility'] < 300:
        score += 6
    elif row['visibility'] < 700:
        score += 2
    elif row['visibility'] < 1000:
        score += 1
    if row['traffic_density'] > 80:
        score += 2
    elif row['traffic_density'] > 60:
        score += 1
    if row['risk_zone'] == 1:
        score += 1
    if score >= 6:
        return 2  # Danger
    elif score >= 3:
        return 1  # Caution
    else:
        return 0  # Safe

df['label'] = df.apply(label_risk, axis=1)

# ------------------- 3. Prepare Features -------------------
X = df[['temperature', 'visibility', 'traffic_density', 'vehicle_speed', 'risk_zone']]
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------------- 4. Train Multiple Models -------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm
    }

    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ------------------- 5. Choose Best Model -------------------
best_model_name = max(results, key=lambda x: results[x]["accuracy"])
best_model = results[best_model_name]["model"]

print(f"\n✅ Best Model: {best_model_name} with Accuracy {results[best_model_name]['accuracy']:.4f}")

# ------------------- 6. Save Best Model -------------------
joblib.dump(best_model, 'risk_predictor8.pkl')
joblib.dump(scaler, 'scaler8.pkl')
print("📦 Best model and scaler saved!")

print(df['label'].value_counts())


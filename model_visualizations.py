import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay
)

import joblib

# Load data
conn = sqlite3.connect('road_safety.db')
df = pd.read_sql_query("SELECT * FROM inputs", conn)
conn.close()

X = df[['temperature', 'visibility', 'traffic_density', 'vehicle_speed', 'risk_zone']]
y = df['label']

#Histogram of features
for col in X:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Boxplots by label
for col in X:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='label', y=col, data=df, palette='Set1')
    plt.title(f'{col} by Risk Level (label)')
    plt.xlabel('Risk Level (0=Safe, 1=Caution, 2=Danger)')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# Scatterplot examples
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='vehicle_speed', y='visibility', hue='label', palette='dark')
plt.title("Speed vs Visibility colored by Risk")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df[['temperature', 'visibility', 'traffic_density', 'vehicle_speed', 'risk_zone', 'label']].corr(), annot=True, cmap='YlGnBu')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Label count plot
plt.figure(figsize=(5, 4))
sns.countplot(x='label', data=df, palette='Set1')
plt.title("Label Distribution")
plt.xlabel("Risk Level (0=Safe, 1=Caution, 2=Danger)")
plt.tight_layout()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "SVM": SVC(kernel='rbf', probability=True)
}

results = []
best_model = None
best_accuracy = 0
best_model_name = ""




for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "name": name,
        "accuracy": acc,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1-score": report["weighted avg"]["f1-score"],
        "cm": cm
    })

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# Save best model
joblib.dump(best_model, 'risk_predictor_test.pkl')
print(f"✅ Best Model: {best_model_name} saved as risk_predictor_test.pkl")

# --- PLOTS ---

# 1. Bar chart: Accuracy comparison
plt.figure(figsize=(8, 5))
sns.barplot(data=pd.DataFrame(results), x='name', y='accuracy', hue='name', palette='Set2', legend=False)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 2. Bar chart: F1-Score comparison
plt.figure(figsize=(8, 5))
sns.barplot(data=pd.DataFrame(results), x='name', y='f1-score', hue='name', palette='Set3', legend=False)
plt.title("Model F1 Score Comparison")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 3. Confusion Matrices
for r in results:
    plt.figure(figsize=(5, 4))
    sns.heatmap(r['cm'], annot=True, fmt="d", cmap='Blues',
                xticklabels=["Safe", "Caution", "Danger"],
                yticklabels=["Safe", "Caution", "Danger"])
    plt.title(f"Confusion Matrix: {r['name']}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

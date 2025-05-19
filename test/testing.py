import joblib
import pandas as pd
import numpy as np

model = joblib.load('risk_predictor_1.pkl')

manual_inputs = [
    {'temperature': 30, 'visibility': 9000, 'traffic_density': 20, 'vehicle_speed': 30, 'risk_zone': 0},  # Should predict Safe
    {'temperature': 35, 'visibility': 4000, 'traffic_density': 60, 'vehicle_speed': 71, 'risk_zone': 0},  # Should predict Caution
    {'temperature': 28, 'visibility': 800, 'traffic_density': 85, 'vehicle_speed': 100, 'risk_zone': 1},  # Should predict Danger
]

# Convert to DataFrame
manual_df = pd.DataFrame(manual_inputs)

predictions = model.predict(manual_df)

# Map predictions to labels
label_map = {0: "Safe", 1: "Caution", 2: "Danger"}

# Display results
for idx, pred in enumerate(predictions):
    print(f"Test case {idx+1}: Prediction = {label_map[pred]}")

print(np.unique(predictions, return_counts=True))





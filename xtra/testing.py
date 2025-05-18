import joblib
import pandas as pd
import numpy as np

# Load your trained model
model = joblib.load('risk_predictor_test.pkl')

# Create manual test cases
manual_inputs = [
    {'temperature': 30, 'visibility': 9000, 'traffic_density': 20, 'vehicle_speed': 30, 'risk_zone': 0},  # Should predict Safe
    {'temperature': 35, 'visibility': 4000, 'traffic_density': 60, 'vehicle_speed': 71, 'risk_zone': 0},  # Should predict Caution
    {'temperature': 28, 'visibility': 800, 'traffic_density': 85, 'vehicle_speed': 100, 'risk_zone': 1},  # Should predict Danger
]

# Convert to DataFrame
manual_df = pd.DataFrame(manual_inputs)

# Predict
predictions = model.predict(manual_df)

# Map predictions to labels
label_map = {0: "Safe", 1: "Caution", 2: "Danger"}

# Display results
for idx, pred in enumerate(predictions):
    print(f"Test case {idx+1}: Prediction = {label_map[pred]}")

print(np.unique(predictions, return_counts=True))

'''import requests

lat, lon = 28.193112781030408,76.94340897627585  # Example coords
api_key = "aed9a7e23f0bb21032ff0304a35175a8"

url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
response = requests.get(url).json()

print(response)  # 🔍 '''




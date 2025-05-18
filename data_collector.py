
import sqlite3
import requests
import random
import csv
from datetime import datetime

# ---------- CONFIG ----------
API_KEY = 'aed9a7e23f0bb21032ff0304a35175a8'  # Replace this
LATITUDE = 22.0576
LONGITUDE = 88.1090
DB_NAME = 'road_safety.db'

# ---------- SETUP DATABASE ----------
def setup_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            latitude REAL,
            longitude REAL,
            temperature REAL,
            visibility INTEGER,
            traffic_density INTEGER,
            vehicle_speed INTEGER,
            risk_zone INTEGER
        )
    ''')
    conn.commit()
    conn.close()

#--------- IMPORTING LOcATIONS ----------

def load_locations_from_csv(file_path):
    locations = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lat = float(row["lat"])
            lon = float(row["lon"])
            locations.append((lat, lon))
    return locations

# ---------- FETCH WEATHER DATA ----------
def fetch_weather(lat, lon, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={'aed9a7e23f0bb21032ff0304a35175a8'}&units=metric"
    response = requests.get(url)


    try:
        data = response.json()
        if "main" not in data:
            print("Error from API:", data)
            return None, None
        temperature = data['main']['temp']
        weather = data['weather'][0]['main'].lower()  # e.g. 'clear', 'rain', 'fog'
        

        # Simulate visibility based on weather condition
        if "fog" in weather or "mist" in weather or "haze" in weather:
            visibility = random.randint(50, 300)  # very low
        elif "rain" in weather or "drizzle" in weather:
            visibility = random.randint(300, 2000)
        elif "cloud" in weather:
            visibility = random.randint(3000, 7000)
        else:
            visibility = random.randint(7000, 10000)  # clear or sunny

        
        return temperature, visibility
    except Exception as e:
        print("Exception while parsing weather data:", e)
        return None, None
    



# ---------- SIMULATE TRAFFIC AND VEHICLE DATA ----------
def fetch_traffic_density(lat=None, lon=None):
    hour = datetime.now().hour

    if 8 <= hour <= 11 or 17 <= hour <= 20:
        # Peak hours
        traffic = random.randint(60, 100)
    elif 12 <= hour <= 16:
        # Moderate
        traffic = random.randint(40, 70)
    else:
        # Low traffic
        traffic = random.randint(20, 40)

    return traffic

def get_vehicle_speed():
    return random.randint(20, 100)

def is_in_risk_zone(lat, lon):
    high_risk_lat = 22.0576
    high_risk_lon = 88.1090
    r_zone=random.choice([0,0,0,1])
    return r_zone
    #return 1 if abs(lat - high_risk_lat) < 0.05 and abs(lon - high_risk_lon) < 0.05 else 0

# Risk Labeling Function
def label_risk(vehicle_speed, visibility, traffic_density, risk_zone):
    score = 0
    if vehicle_speed > 90:
        score += 6
    elif vehicle_speed > 70:
        score += 3
    elif vehicle_speed > 50:
        score += 1
    if visibility < 300:
        score += 6
    elif visibility < 700:
        score += 2
    elif visibility < 1000:
        score += 1
    if traffic_density > 80:
        score += 2
    elif traffic_density > 60:
        score += 1
    if risk_zone == 1:
        score += 1
    if score >= 6:
        return 2  # Danger
    elif score >= 3:
        return 1  # Caution
    else:
        return 0  # Safe
    
    



# ---------- SAVE TO DATABASE ----------
def save_to_db(lat, lon, temperature, visibility, traffic_density, speed, risk_zone,label):
    #label = label_risk(speed, visibility, traffic_density, risk_zone)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO inputs (timestamp, latitude, longitude, temperature, visibility, traffic_density, vehicle_speed, risk_zone, label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, lat, lon, temperature, visibility, traffic_density, speed, risk_zone,label))
    conn.commit()
    conn.close()

# ---------- MAIN EXECUTION ----------
def main():
    setup_database()

    # Define multiple random locations
    locations = load_locations_from_csv("indian_1000_cities.csv")

    for i in range(1):
        lat, lon = random.choice(locations)
        temperature, visibility = fetch_weather(lat, lon, API_KEY)
        if temperature is None:
            print("Failed to fetch weather data. Skipping this entry.")
            continue
        traffic_density = fetch_traffic_density()
        speed = get_vehicle_speed()
        risk_zone = is_in_risk_zone(lat, lon)
        label = label_risk(speed, visibility, traffic_density, risk_zone)
        save_to_db(lat, lon, temperature, visibility, traffic_density, speed, risk_zone,label)
        print(f"Data from {lat}, {lon} collected and saved successfully.")



if __name__ == '__main__':
    main()

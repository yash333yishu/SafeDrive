from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import joblib
import numpy as np
import pandas as pd
import requests
import random
from datetime import datetime,timedelta
from flask import Flask, render_template, request
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import os




app = Flask(__name__)
model = joblib.load('risk_predictor_1.pkl')

app.secret_key = 'your-secret-key'  # Needed for session
app.permanent_session_lifetime = timedelta(days=1)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")



# Geocode address if lat/lon is not given
def geocode_address(address):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={GOOGLE_API_KEY}"
    data = requests.get(url).json()
    if data['status'] == 'OK':
        loc = data['results'][0]['geometry']['location']
        return loc['lat'], loc['lng']
    return None, None

# Reverse geocode lat/lon to address
def reverse_geocode(lat, lon):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={GOOGLE_API_KEY}"
    data = requests.get(url).json()
    if data['status'] == 'OK':
        return data['results'][0]['formatted_address']
    return None

# Weather fetch
def fetch_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    data = requests.get(url).json()
    try:
        temperature = data['main']['temp']
        condition = data['weather'][0]['main'].lower()
        

        # Simulate visibility
        if "fog" in condition or "mist" in condition or "haze" in condition:
            visibility = random.randint(50, 500)
        elif "rain" in condition or "drizzle" in condition:
            visibility = random.randint(1000, 4000)
        elif "cloud" in condition or "overcast" in condition:
            visibility = random.randint(6000, 9000)
        else:
            visibility = random.randint(9000, 10000)

        return temperature, visibility, condition.capitalize()

    except (KeyError, IndexError) as e:
        print(f"[Error] Could not parse weather data: {e}")
        return None, None, "Unknown"
    

# Check if inside a known risk zone
def is_in_risk_zone(lat, lon):
    if lat is None or lon is None:
        return 0
    risky_locations = [
        (28.6139, 77.2090)   
    ]
    for rlat, rlon in risky_locations:
        if abs(lat - rlat) < 0.05 and abs(lon - rlon) < 0.05:
            return 1
    return 0


def get_traffic_density_by_time():
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

@app.route("/", methods=["GET", "POST"])
def login():

    if "user" in session:
        return redirect(url_for("predict_page"))  #  Redirect if already logged in
    
    if request.method == "POST":
        action = request.form.get("action")
        username = request.form.get("username")
        password = request.form.get("password")

        conn = sqlite3.connect("road_safety.db")
        cursor = conn.cursor()

        if action == "Login":
            cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            if row and check_password_hash(row[0], password):
                session["user"] = username
                session.permanent = True  # ✅ this keeps the session across tabs/browser close

                return redirect(url_for("predict_page"))
            else:
                flash("Invalid credentials")

        elif action == "Register":
            try:
                if len(password) < 5:
                    flash("Password must be at least 5 characters.")
                else:
                    hashed_pw = generate_password_hash(password)
                    cursor.execute("INSERT INTO users (username, name, password) VALUES (?, ?, ?)", (username, request.form.get("name"), hashed_pw))
                    conn.commit()
                    flash("Registration successful! Please log in.")

                conn.commit()
                flash("Registration successful! Please log in.")
            except sqlite3.IntegrityError:
                flash("Username already exists.")

        conn.close()

    return render_template("login.html")

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user", None)  # Clear user session
    return redirect(url_for("login"))  # Redirect to login page



@app.route("/predict", methods=["GET", "POST"])
def predict_page():

    if "user" not in session:
        return redirect(url_for("login"))  # 🔒 Protect the page
    # Fetch recent data for weather snapshot
    conn = sqlite3.connect("road_safety.db")
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, temperature, visibility FROM inputs ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()


    if row:
        timestamps = [row[0]]
        temperatures = [row[1]]
        visibility_vals = [row[2]]
    else:
        timestamps = []
        temperatures = []
        visibility_vals = []

    prediction = recommendation = None
    lat = lon = level = None
    address_name = None
    visibility_label = None
    weather = "Unavailable"
    current_time = datetime.now().strftime("%d %b %Y, %I:%M %p")


    if request.method == "POST":
        address = request.form['address']
        speed = int(request.form['vehicle_speed'])
        health = request.form['driver_health']
        lat_form = request.form.get("latitude")
        lon_form = request.form.get("longitude")

        # Use provided lat/lon or geocode
        if lat_form and lon_form:
            lat = float(lat_form)
            lon = float(lon_form)
        else:
            lat, lon = geocode_address(address)

        if lat is None or lon is None:
            prediction = "Invalid address or location."
        else:
            temperature, visibility,weather = fetch_weather(lat, lon)
            traffic = get_traffic_density_by_time()
            risk_zone = is_in_risk_zone(lat, lon)

            visibility_vals = [visibility]
            temperatures = [temperature]
            timestamps = [datetime.now().isoformat()]


            # Health risk impact
            health_risk_map = {
                "Healthy": 0,
                "Fatigued / Drowsy": 1,
                "Vision Impaired": 1,
                "Under Medication": 1,
                "Unwell (e.g. fever, dizziness)": 1
            }
            health_risk = health_risk_map.get(health, 0)

            # Optionally adjust traffic/speed
            if health_risk == 1:
                traffic += 10
                speed += 5

            features = pd.DataFrame([{
                'temperature': temperature,
                'visibility': visibility,
                'traffic_density': traffic,
                'vehicle_speed': speed,
                'risk_zone': risk_zone
            }])

            label = int(model.predict(features)[0])
            label_map = {0: "Safe", 1: "Caution", 2: "Danger"}

            # Escalate due to driver health
            if health_risk == 1 and label < 2:
                label += 1

            level = label_map[label]

            def get_custom_recommendation(level, health, visibility, speed):
                vis_label = "Low" if visibility <= 300 else "Moderate" if visibility <= 7000 else "Clear"
                
                # Base suggestions
                if level == "Safe":
                    rec = "Maintain current speed and stay attentive."
                elif level == "Caution":
                    rec = "Slow down, stay alert, and avoid risky maneuvers."
                else:
                    rec = "Avoid travel if possible. Drive slowly with hazards on."

                # Health-specific
                if health != "Healthy":
                    if level == "Safe":
                        rec += " However, monitor your health closely and avoid long drives."
                    elif level == "Caution":
                        rec += " Since you're not feeling well, avoid highways or heavy traffic zones."
                    elif level == "Danger":
                        rec += " Driving in your current health condition is strongly discouraged."

                # Visibility impact
                if vis_label == "Low":
                    rec += " Visibility is low — use fog lights, reduce speed, and maintain wide gaps."
                elif vis_label == "Moderate" and level != "Safe":
                    rec += " Visibility is moderate — stay cautious near junctions."

                # Speeding
                if speed > 70 and level != "Safe":
                    rec += " Your current speed is risky for these conditions — reduce to below 50 km/h."

                # Final 
                rec = rec.strip().replace("  ", " ")
                return rec

            recommendation = get_custom_recommendation(level, health, visibility, speed)


            # Log data
            timestamp = datetime.now().isoformat()
            conn = sqlite3.connect("road_safety.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO inputs (timestamp, latitude, longitude, temperature, visibility, traffic_density, vehicle_speed, risk_zone, label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, lat, lon, temperature, visibility, traffic, speed, risk_zone,label))
            conn.commit()
            conn.close()

            prediction = f"{level} at {reverse_geocode(lat, lon) or f'{lat}, {lon}'}"

            address_name = reverse_geocode(lat, lon)

                # Categorize visibility
            if visibility <= 300:
                visibility_label = "Low"
            elif visibility <= 7000:
                visibility_label = "Moderate"
            else:
                visibility_label = "Clear"

            
            



    return render_template("predict.html",prediction=prediction,
                            recommendation=recommendation,lat=lat,lon=lon,level=level,
                            address=address_name,timestamps=timestamps,temperatures=temperatures,weather=weather,
                            visibility=visibility_vals,visibility_label=visibility_label,current_time=current_time,google_api_key=GOOGLE_API_KEY)


if __name__ == "__main__":
    app.run(debug=True)

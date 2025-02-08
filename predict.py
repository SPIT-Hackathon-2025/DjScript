import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from datetime import datetime

# Load environment variables
load_dotenv()
google_api_key = os.getenv('apiKey')

# Initialize Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

app = Flask(__name__)
CORS(app)

def calculate_idle_time(last_service_date: str, km_driven: int, expected_daily_usage: int = 30) -> int:
    """
    Calculates the idle time of the bike in days.
    
    :param last_service_date: Date of last service (YYYY-MM-DD).
    :param km_driven: Total kilometers traveled since last service.
    :param expected_daily_usage: Average daily distance traveled (default: 30 km).
    :return: Estimated idle time in days.
    """
    today = datetime.today()
    last_service = datetime.strptime(last_service_date, "%Y-%m-%d")
    
    days_since_service = (today - last_service).days
    active_days = km_driven // expected_daily_usage  # Estimate how many days it was used
    
    idle_time = max(0, days_since_service - active_days)  # Ensure it doesn't go negative
    return idle_time

def predict_maintenance(km_driven, last_service_date):
    """Predict maintenance score and battery health using predefined variables."""
    riding_history = [
        "Rough terrain", "Smooth road", "Gravel path", "Wet road",
        "Hilly terrain", "Paved road", "Off-road", "City streets",
        "Uneven pavement", "Muddy track"
    ]  # Last 10 ride terrains

    battery_charge_cycles = 10000
    idle_time = calculate_idle_time(last_service_date, km_driven)
    charging_history = "Regular overnight charging"

    prompt = f"""
    You are an expert in electric bicycle maintenance and predictive diagnostics.
    Given the following conditions:
    - Riding History (Last 10 terrains): {riding_history}
    - Last Service Date: {last_service_date}
    - Kilometers Driven: {km_driven}
    - Battery Charge Cycles: {battery_charge_cycles}
    - Idle Time (days): {idle_time}
    - Charging History: {charging_history}

    Predict:
    1. Battery Health Score (1-100, higher is better).
    2. Maintenance Risk Score (1-10, 10 is high risk).
    3. Replacement of battery Needed? (0 = No, 1 = Yes).
    4. Maintenance Needed? (0 = No, 1 = Yes).

    **Strict JSON Response Format:**
    ```json
    {{
        "battery_score": <integer>,
        "maintenance_score": <integer>,
        "replacement_needed": <0 or 1>,
        "maintenance_needed": <0 or 1>
    }}
    ```
    """

    # Define JSON response structure
    battery_score_schema = ResponseSchema(name="battery_score", description="Battery health score (1-100).")
    maintenance_score_schema = ResponseSchema(name="maintenance_score", description="Maintenance risk score (1-10).")
    replacement_schema = ResponseSchema(name="replacement_needed", description="0 = No, 1 = Yes for battery replacement.")
    maintenance_schema = ResponseSchema(name="maintenance_needed", description="0 = No, 1 = Yes for service needed.")

    response = gemini_model.invoke(prompt)
    output_parser = StructuredOutputParser.from_response_schemas([
        battery_score_schema, maintenance_score_schema, replacement_schema, maintenance_schema
    ])
    output = output_parser.parse(response.content)
    return output

@app.route('/predict-maintenance', methods=['POST'])
def maintenance_api():
    """API endpoint to predict maintenance needs."""
    data = request.json

    km_driven = data.get("km_driven")
    last_service_date = data.get("last_service_date")

    if not km_driven or not last_service_date:
        return jsonify({"error": "Both kilometers driven and last service date are required"}), 400

    predictions = predict_maintenance(km_driven, last_service_date)

    return jsonify({
        "battery_score": predictions["battery_score"],
        "maintenance_score": predictions["maintenance_score"],
        "replacement_needed": predictions["replacement_needed"],
        "maintenance_needed": predictions["maintenance_needed"]
    })

if __name__ == '__main__':
    app.run(debug=True)

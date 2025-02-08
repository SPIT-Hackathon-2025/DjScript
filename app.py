import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app) 
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Load environment variables
load_dotenv()
google_api_key = os.getenv('apiKey') 
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") 

# Initialize Gemini model with the API key
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

from datetime import datetime

def get_time_and_season():
    # Get current hour and month
    now = datetime.now()
    hour = now.hour
    month = now.month

    # Determine day or night
    if 6 <= hour < 18:
        time_of_day = "Day"
    else:
        time_of_day = "Night"

    # Determine season (Northern Hemisphere logic)
    if month in [12, 1, 2]:
        season = "Winter"
    elif month in [3, 4, 5,6]:
        season = "Summer"
    else:
        season = "Rainy"

    return f"{time_of_day}, {season} season"

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            # Extract weather details
            temperature = data["main"]["feels_like"]
            weather_description = data["weather"][0]["description"]
            wind_speed = data["wind"]["speed"]
            humidity = data["main"]["humidity"]
            visibility = data["visibility"]

            # Format the weather string 
            weather_info = f"{weather_description.capitalize()}, {temperature}°C, Wind {wind_speed} m/s, Humidity {humidity}, visibility {visibility}%"
            return weather_info

        else:
            return f"Error: {data['message']}"

    except Exception as e:
        return f"Error: {str(e)}"



def predict_risk(weather, time,sloc,des):
    prompt = f"""
    You are the best risk predictor and you have great knowledge about cycling.
    Given the following conditions:
    - Terrain: Figure out terrain based on starting location {sloc} to destination {des}
    - Weather: {weather}
    - Time: {time}

    1 Predict the risk score on a scale of 1 to 10, where 1 is minimal risk and 10 is extreme risk to be faced by the rider.
    2 Provide a short explanation for the score in readable format.
    3 Also provide 5 adaptive ride strategies to enhance cyclist/ 2 wheeler safety.
    **Response Format (Strictly JSON no extra text):**
    ```json
    {{
        "risk_score": <integer>,
        "reason": "<short explanation>",
        "suggestions": "<safety recommendations>"
    }}
    ```
    """
    risk_score_schema = ResponseSchema(name="risk_score", description="Risk score from 1 to 10, where 1 is low risk and 10 is high risk.")
    reason_schema = ResponseSchema(name="reason", description="Short explanation for the assigned risk score.")
    suggestions_schema = ResponseSchema(name="suggestions", description="List of safety tips or precautions.")
    response = gemini_model.invoke(prompt)
    output_parser= StructuredOutputParser.from_response_schemas([risk_score_schema,reason_schema,suggestions_schema])
    output = output_parser.parse(response.content)
    return output



@app.route('/predict-risk', methods=['POST'])
def risk_prediction_api():
    data = request.json
    sloc = data.get("starting_location")
    des = data.get("destination")
    
    if not sloc or not des:
        return jsonify({"error": "Both starting location and destination are required"}), 400
    
    weather = get_weather(des)  # Get weather for the starting location
    time = get_time_and_season()
    
    risk_score = predict_risk(sloc, des, weather, time)
    
    return jsonify({
        "starting_location": sloc,
        "destination": des,
        "weather": weather,
        "time": time,
        "risk_score": risk_score
    })

if __name__ == '__main__':
    app.run(debug=True)

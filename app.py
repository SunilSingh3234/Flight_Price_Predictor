from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Encoding dictionaries
airline_dict = {'AirAsia': 0, "Indigo": 1, "GO_FIRST": 2, "SpiceJet": 3, "Air_India": 4, "Vistara": 5}
source_dict = {'Delhi': 0, "Hyderabad": 1, "Bangalore": 2, "Mumbai": 3, "Kolkata": 4, "Chennai": 5}
departure_dict = {'Early_Morning': 0, "Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4, "Late_Night": 5}
stops_dict = {'zero': 0, "one": 1, "two_or_more": 2}
arrival_dict = {'Early_Morning': 0, "Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4, "Late_Night": 5}
destination_dict = {'Delhi': 0, "Hyderabad": 1, "Mumbai": 2, "Bangalore": 3, "Chennai": 4, "Kolkata": 5}
class_dict = {'Economy': 0, 'Business': 1}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        airline = airline_dict[data['airline']]
        source_city = source_dict[data['source_city']]
        departure_time = departure_dict[data['departure_time']]
        stops = stops_dict[data['stops']]
        arrival_time = arrival_dict[data['arrival_time']]
        destination_city = destination_dict[data['destination_city']]
        travel_class = class_dict[data['class']]

        # Date difference
        departure_date = datetime.strptime(data['departure_date'], "%Y-%m-%d")
        today = datetime.today()
        date_diff = (departure_date - today).days + 1

        # Prepare input
        input_features = [[
            airline, source_city, departure_time, stops,
            arrival_time, destination_city, travel_class, date_diff
        ]]

        # Predict
        prediction = model.predict(input_features)[0]

        return jsonify({'prediction': round(prediction, 2)})
    except KeyError as e:
        return jsonify({'error': f'Missing or wrong input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

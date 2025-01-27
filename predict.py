import pickle
import pandas as pd
from flask import Flask, request, jsonify
import xgboost as xgb


# Load the model and DictVectorizer
model_file = 'xgb_model_trained.pkl'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('bike_demand_prediction')

@app.route('/predict', methods=['POST'])


def predict():
    print('..........Predition function started........')
    test_data = request.get_json()

    # Validate input (optional, but recommended)
    if not test_data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # Transform input data using the DictVectorizer
        X_test = dv.transform([test_data])

        # Make prediction
        y_pred = model.predict(X_test)[0]  # Get the predicted value

        # Prepare the result
        result = {
            'predicted_bike_demand': int(round(y_pred))
        }

        #print(f"Input data: {test_data}")
        #print(f"Prediction: {y_pred}")
        return jsonify(result)

    except Exception as e:
        # Handle any exceptions during prediction
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error during prediction', 'details': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
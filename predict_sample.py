import requests
import json

print('..........Starting processing.......')

url = "http://127.0.0.1:9696/predict"

# Sample input data
test_data = {'hour': 8, 
             'temperature_c': 18.2, 
             'humidity': 63,
              'wind_speed_ms': 0.8, 
              'visibility_10m': 1731, 
              'dew_point_temperature_c': 11.0, 
              'solar_radiation_mjm2': 1.0, 
              'rainfallmm': 0.0, 
              'snowfall_cm': 0.0, 
              'day': 22, 'month': 5, 
              'year': 2018, 
              'seasons_Autumn': False, 
              'seasons_Spring': True, 
              'seasons_Summer': False, 
              'seasons_Winter': False, 
              'holiday_Holiday': True, 
              'holiday_No Holiday': False, 
              'functioning_day_No': False, 
              'functioning_day_Yes': True}


# Send request to the prediction endpoint
try:
    print('........Sending request..........')
    response = requests.post(url, json=test_data, timeout=10).json()
    bike_demand = response['predicted_bike_demand']
    print(f'Predicted amount of rented bikes:{bike_demand} pcs')

except requests.exceptions.ConnectionError as e:
    print(f"Error: Could not connect to the server at {url}. Please make sure the server is running and the URL is correct.")
    print(f"Details: {e}")

except requests.exceptions.Timeout as e:
    print(f"Error: Connection to the server at {url} timed out.")
    print(f"Details: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
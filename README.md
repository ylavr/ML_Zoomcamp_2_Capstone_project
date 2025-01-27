# ML_Zoomcamp_2_Capstone_project (Bike demand prediction)
## Summary

This project aims to predict daily rental bike demand based on environmental and seasonal conditions. 
By leveraging historical data, this regression problem seeks to optimize bike-sharing systems, helping operators make informed decisions regarding resource allocation, inventory management, and system efficiency.
The dataset provided by UCI Machine Learning repository [Bike Sharing Demand Dataset](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand).

The dataset used is the Bike Sharing Demand Dataset, provided by Kaggle.
1.  Problem Description
2.  Dependency and Environment Management
3.  Exploratory Data Analysis (EDA)
4.  Data Preparation
5.  Model Training and Parameters Tuning
6.  Comparing Models' Performance and Selecting the Best Model
7.  Creating Python Scripts from Notebook
8.  Local Model Deployment with Docker

## 1.  Problem Description
Efficient resource allocation is critical for bike-sharing services to meet user demand and avoid overstocking and understocking at docking stations. 
The primary objective of this project is to predict the number of bikes rented per day using features such as temperature, humidity, weather conditions, and more. 
This prediction allows for better system optimization, ensuring users can access bikes when needed.

## 2.  Dependency and Environment Management
This project uses a requirements.txt file for dependency and environment management. To set up the project environment, follow these steps:
1. Clone the repository:   
```
git clone 
git cd bicycle-sharing-demand
``` 
2. Create and activate a virtual environment:
```
python -m .venv env
source .venv/bin/activate # For Linux/macOS
.venv\Scripts\activate # For Windows
```
3. Install project dependencies:
```
pip install -r requirements.txt
```

## 3.  Exploratory Data Analysis (EDA)

The Exploratory Data Analysis (EDA) provides key insights into the dataset:

No Missing and duplicated Values: The dataset is clean and complete, with no missing and duplicated values.

Target Variable Distribution: The target variable, count (number of rented bikes), follows a skewed distribution.

Seasonality: Clear seasonal trends are observed, with higher demand during warmer months.

Feature Correlation: Variables such as temperature and humidity show strong relationships with the target variable, while weather and working day influence demand.

Key visualizations and statistics are included in the EDA - bikes.ipynb file.

## 4.  Data Preparation
In the data preparation phase:

Column names were standardized (lowercase, underscores, and replacing spaces).

The dataset was split into training (60%), validation (20%), and testing (20%) sets.

Categorical features were encoded using one-hot encoding.

## 5.  Model Training and Parameters Tuning
Several machine learning models were trained and tuned:

**Linear Regression**: Baseline model to assess linear relationships.
**Ridge and Lasso Regression**
**Random Forest**: Tuned with n_estimators, max_depth, and min_samples_leaf.
**XGBoost**: Tuned with max_depth, eta, and subsample.

## 6.  Comparing Models' Performance and Selecting the Best Model
Models were evaluated based on RMSE (Root Mean Squared Error) on the validation set. The best-performing model was selected for deployment:

## 7.  Creating Python Scripts from Notebook
Code from Jupyter Notebooks was converted into reusable Python scripts:

train.py: Trains the final model using the full training dataset, saving the model and preprocessing steps into a binary file (xgboost_model.bin).

```
python train.py
```

predict.py: Loads the trained model, serving predictions through a Flask API (/predict endpoint).

## 8.  Local Model Deployment with Docker
The prediction service can be deployed locally using Docker:
1. Build the Docker image:
```
```
2. Run the Docker container:
```
```
4. Test the service using predict_sample.py:
```
```
**Files in the Repository:**


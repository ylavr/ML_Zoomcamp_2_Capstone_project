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

**The Exploratory Data Analysis (EDA)** provides key insights into the dataset:

**No Missing and duplicated Values**: The dataset is clean and complete, with no missing and duplicated values.

**Target Variable Distribution**: The target variable, count (number of rented bikes), follows a skewed distribution.

**Seasonality**: Clear seasonal trends are observed, with higher demand during warmer months.

**Feature Correlation**: Variables such as temperature and humidity show strong relationships with the target variable, while weather and working day influence demand.

**Key visualizations and statistics are included in the EDA - bikes.ipynb file**.

## 4.  Data Preparation
In the data preparation phase:

Column names were standardized (lowercase, underscores, and replacing spaces, removed special signs).

The dataset was split into training (60%), validation (20%), and testing (20%) sets.

Categorical features were encoded using one-hot encoding.

## 5.  Model Training and Parameters Tuning
Several machine learning models were trained and tuned:

**Linear Regression**: Baseline model to assess linear relationships.
**Ridge and Lasso Regression**: Tuned with Alpha value.
**Random Forest**: Tuned with n_estimators, max_depth, and min_samples_leaf,min_samples_split.
**XGBoost**: Tuned withn_estimators, max_depth,learning_rate, min_child_weight, subsample, colsample_bytree.

## 6.  Comparing Models' Performance and Selecting the Best Model
Models were evaluated based on RMSE (Root Mean Squared Error) on the validation set and  R² Score. The best-performing model was selected for deployment:
**XGBoost**
| **Model**           | **Validation RMSE** | **Test RMSE**    | **R² Score**     |
|---------------------|---------------------|------------------|------------------|
| Linear Regression   | 461.8096           | 465.5259         | 0.4756           |
| Ridge Regression    | 461.8165           | 465.5261         | 0.4756           |
| Lasso Regression    | 461.8180           | 465.5252         | 0.4756           |
| Random Forest       | 340.7357           | 327.9956         | 0.7397           |
| XGBoost Regressor*  | 301.0234           | 301.9364         | 0.7794           |


## 7.  Creating Python Scripts from Notebook
Code from Jupyter Notebooks was converted into reusable Python scripts:

train.py: Trains the final model using the full training dataset, saving the model and preprocessing steps into a binary file (xgb_model_trained.pkl).

```
python train.py
```

predict.py: Loads the trained model, serving predictions through a Flask API (/predict endpoint).

```
python predict.py
```
preidct_sample.py: Pass sample data to model for test and getting result:

```
python predict_sample.py
```

## 8.  Local Model Deployment with Docker
The prediction service can be deployed locally using Docker:
1. Build the Docker image:
```
docker build -t bike-demand-prediction .
```
2. Run the Docker container:
```
docker run -it -p 9696:9696 bike-demand-prediction     
```
4. Test the service using predict_sample.py:
```
python predict_sample.py
```
**Files in the Repository:**
**SeoulBikeData.csv**
**EDA - bikes.ipynb**
**train.py**
**xgb_model_trained.pkl**
**predict.py**
**predict_sample.py**
**Dockerfile**
**requirements.txt**
**Screenshot with model output**





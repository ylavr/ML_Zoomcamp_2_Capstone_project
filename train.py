
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
import xgboost
from xgboost import XGBRegressor
import pickle

# XGB parameters:

xgb_params = {'n_estimators': 1000, 
              'learning_rate': 0.05, 
              'max_depth': 10, 
              'min_child_weight': 5, 
              'subsample': 0.7, 
              'colsample_bytree': 1.0}

output_file = 'xgb_model_trained.pkl'

# Data Loading and preparation:
df = pd.read_csv('SeoulBikeData.csv', encoding='ISO-8859-9')

#Data preprocessing:

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df['datetime'] = df['Date'] + pd.to_timedelta(df['Hour'], unit='h')
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)

# Extract features from the date:
df['Day']=df['Date'].dt.day
df['Month']=df['Date'].dt.month
df['Year']=df['Date'].dt.year

# Rename columns:
df = df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))
df.columns = (
    df.columns.str.lower()
    .str.strip()
    .str.replace('[()/%]', '', regex=True)  # remove parentheses, slash, percent
    .str.replace(' ', '_')
    .str.replace('°c', '_c')
    .str.replace('mj/m2', 'mj_m2')
)

# Fix duplicate column names
#df.columns.values[2]  = 'hour'  # the first hour
#print(df.columns)

# One-hot encode categorical columns:
df_encoded = pd.get_dummies(df, 
                            columns=['seasons', 'holiday', 'functioning_day'], 
                            drop_first=False)


# Drop unnecessary columns:
df_encoded.drop('date', axis =1,  inplace = True)


#  #Split the data into 3 parts: train/validation/test with 60%/20%/20% distribution: 
df_train_full, df_test = train_test_split(df_encoded, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)

#  Reset the Dataframe indexes:
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Prepare target variables
y_train = df_train['rented_bike_count'].values
y_val = df_val['rented_bike_count'].values
y_test = df_test['rented_bike_count'].values

# Convert training data to dictionary format for DictVectorizer
X_train_dict = df_train.drop('rented_bike_count', axis=1).to_dict(orient='records')
X_val_dict = df_val.drop('rented_bike_count', axis=1).to_dict(orient='records')
X_test_dict = df_test.drop('rented_bike_count', axis=1).to_dict(orient='records')

# Initialize and fit DictVectorizer
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(X_train_dict)
X_val = dv.transform(X_val_dict)
X_test = dv.transform(X_test_dict)

# Initialize the model with the current parameters:
xgb = XGBRegressor(**xgb_params)
xgb.fit(X_train, y_train)

# Evaluate the model
val_rmse = root_mean_squared_error(y_val, xgb.predict(X_val))
print(f'Validation RMSE: {val_rmse}')

 # Save the model and DictVectorizer
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, xgb), f_out)

print(f'The trained model and DictVectorizer are saved to {output_file}')

# Prepare a sample test data for validation
test_data_dict = X_test_dict[0]
print(f'Sample test data formatted for prediction: {test_data_dict}')


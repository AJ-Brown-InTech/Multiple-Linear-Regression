import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ^y=a+bx 

# Load the training & test data
df = pd.read_csv('train.csv')

# Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Fill missing values in numeric columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Features and target variable for training
X_train = df[numeric_cols].drop(columns='SalePrice', errors='ignore')
y_train = df['SalePrice']

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Load the test data
dft = pd.read_csv('test.csv')

# Ensure the test data has no missing values
t_cols = dft.select_dtypes(include=[np.number]).columns
X_test = dft[t_cols].fillna(dft[t_cols].mean())

# Check if 'SalePrice' is in the test set
if 'SalePrice' in dft.columns:
    y_test = dft['SalePrice'].fillna(dft['SalePrice'].mean())

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
else:
    # 'SalePrice' is not in the test set
    print("SalePrice column is not present in the test data.")
    y_pred = model.predict(X_test)
    print("Predictions on the test set:")
    print(y_pred)

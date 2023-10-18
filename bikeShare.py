# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Reading the data
df = pd.read_csv('day.csv')

# Data Exploration
df.head()

# Data Pre-processing(cleaning)
# Converting 'season' and 'weathersit' columns to string type for proper categorical representation
df['season'] = df['season'].map({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})
df['weathersit'] = df['weathersit'].map({
    1: 'clear',
    2: 'mist',
    3: 'light_snow_rain',
    4: 'heavy_rain_ice'
})

# Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns

#print all libraries used for documentation
print("pandas version:"+pd.__version__)
print("numpy version:"+np.__version__)
print("seaborn version:"+sns.__version__)

# Plot pairwise relationships in a dataset
sns.pairplot(df[['temp', 'atemp', 'hum', 'windspeed', 'cnt']])
plt.show()

# Creating dummy variables for categorical columns
df = pd.get_dummies(df, columns=['season', 'weathersit'], drop_first=True)

# Splitting the dataset

X = df.drop(columns=['instant', 'dteday', 'casual', 'registered', 'cnt'])
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

# Model Evaluation
r2 = r2_score(y_test, y_pred)
print(f"R-squared value on the test set: {r2}")

# Coefficients and feature importance
coefficients = pd.DataFrame({
    'Features': X_train.columns,
    'Coefficients': lm.coef_
}).sort_values(by='Coefficients', ascending=False)

print(coefficients)

# Residual analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.displot(residuals)
plt.title('Residual Analysis')
plt.xlabel('Residuals')
plt.show()

# Checking actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()


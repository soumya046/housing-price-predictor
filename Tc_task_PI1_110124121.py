import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df = pd.read_csv("C:/Users/91963/Downloads/California_Houses - California_Houses.csv")


# Display basic information about the dataset
print("Dataset Shape:", df.shape) 
print("\nColumn Names:")  
print(df.columns.tolist())
print("\nFirst 5 rows:")
df.head(5)


# Basic information about the dataset
print("Dataset Info:")
print(df.info()) 
print("\nData Types:")
print(df.dtypes)
print("\nDescriptive Statistics:")
df.describe() 

        

# Visualize median value distribution
sns.histplot(df['Median_House_Value']) 
plt.show()



# Check for missing values
print("Missing Values in Tot_Bedrooms:")
print(df['Tot_Bedrooms'].isnull().sum())
print("Number of non-null entries in 'Tot_Bedrooms':", df['Tot_Bedrooms'].dropna().shape[0])



# Handling missing values in 'Tot_Bedrooms' column by replacing NaNs with the median (if possible)
if df['Tot_Bedrooms'].dropna().shape[0] > 0:
    median_bed = df['Tot_Bedrooms'].median()
    print("Median value of 'Tot_Bedrooms':", median_bed)

    # Fill all NaN (missing) values in 'Tot_Bedrooms' with the calculated median
    df['Tot_Bedrooms'] = df['Tot_Bedrooms'].fillna(median_bed)

else:    
    print("Warning: 'Tot_Bedrooms' has no valid (non-NaN) entries to compute median.")
    

mean_value = df['Median_House_Value'].mean()
df['mean_house_value'] = mean_value


from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = df.drop(['Median_House_Value', 'mean_house_value'], axis=1)
y_median = df['Median_House_Value']
y_mean = df['mean_house_value']

# Splitting into training and test sets (80% train, 20% test)
X_train_median, X_test_median, y_train_median, y_test_median = train_test_split(X, y_median, test_size=0.2, random_state=42)
X_train_mean, X_test_mean, y_train_mean, y_test_mean = train_test_split(X, y_mean, test_size=0.2, random_state=42)
    


# Training a Linear Regression model and making predictions on test data

from sklearn.linear_model import LinearRegression

model_median = LinearRegression()
model_median.fit(X_train_median, y_train_median)

model_mean = LinearRegression()
model_mean.fit(X_train_mean, y_train_mean)

y_pred_median = model_median.predict(X_test_median)
print("First 5 predicted values of 'Median_House_Value':", y_pred_median[:5])
y_pred_mean = model_mean.predict(X_test_mean)
print("First 5 predicted values of 'mean_house_value':", y_pred_mean[:5])





# Evaluating model performance using MAE, MSE, and RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Median Model:")
print("MAE:", mean_absolute_error(y_test_median, y_pred_median))
print("MSE:", mean_squared_error(y_test_median, y_pred_median))
print("RMSE:", np.sqrt(mean_squared_error(y_test_median, y_pred_median)))

print("\nMean Model:")
print("MAE:", mean_absolute_error(y_test_mean, y_pred_mean))
print("MSE:", mean_squared_error(y_test_mean, y_pred_mean))
print("RMSE:", np.sqrt(mean_squared_error(y_test_mean, y_pred_mean)))


# Visualizing the relationship between actual and predicted house values
plt.figure(figsize=(8, 6))  # Set figure size

plt.scatter(y_test_median, y_pred_median, alpha=0.5, color='teal', edgecolors='w', linewidths=0.5)
plt.plot([y_test_median.min(), y_test_median.max()], [y_test_median.min(), y_test_median.max()], 'r--', lw=2)  

plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs Predicted House Values")
plt.grid(True)
plt.tight_layout()
plt.show()


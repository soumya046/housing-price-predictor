# housing-price-predictor
ML model to predict mean and median house values

This project uses machine learning to predict the **mean and median housing values** based on various housing features like Tot_Bedrooms, population, etc.

---

## Dataset

The dataset was provided via Google Sheets and contains the following features:

- `Longitude`, `Latitude`
- `Housing_Age`
- `Tot_Rooms`, `Tot_Bedrooms`
- `Population`, `Households`
- `Median_Income`
- `Median_House_Value` (Target variable)

---

##  Objective

Build and train a regression model to:

- Predict **Median House Value**
-  Predict **Mean House Value**

---

##  Machine Learning Techniques Used

- Data Preprocessing (`pandas`, `numpy`)
- Missing Value Handling (using **median imputation**)
- Data Splitting (`train_test_split`)
- Model: **Linear Regression**
- Evaluation Metrics:
  - **MAE** – Mean Absolute Error
  - **MSE** – Mean Squared Error
  - **RMSE** – Root Mean Squared Error

---

## Results

- The model was trained using **80%** of the data and tested on **20%**.
- Prediction results were evaluated using MAE, MSE, and RMSE.

##  How to Run

1. Clone the repository: https://github.com/soumya046/housing-price-predictor.git
2. Open the project in your IDE (e.g., Spyder or VS Code)
3. Run the script: python Tc_task_PI1_110124121.py 
4. Make sure required libraries are installed: pandas,numpy, matplotlib, seaborn, scikitlearn
5. That's it!





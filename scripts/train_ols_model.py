import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import os
import pickle

# Step 1: Load the training and test data
train_data_path = os.path.join("data", "trainingdata.csv")
test_data_path = os.path.join("data", "testdata.csv")

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Step 2: Prepare features and target
X_train = train_data.drop(columns=["Severity"])
y_train = train_data["Severity"]

X_test = test_data.drop(columns=["Severity"])
y_test = test_data["Severity"]

# Drop irrelevant columns and keep numeric ones
columns_to_drop = ["P_CODE", "VDC_NAME", "DISTRICT", "REGION", "Severity category"]
X_train = X_train.drop(columns=columns_to_drop, errors="ignore")
X_test = X_test.drop(columns=columns_to_drop, errors="ignore")

# Add a constant for the intercept
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Step 3: Train the OLS model
ols_model = sm.OLS(y_train, X_train).fit()

# Step 4: Evaluate the model
y_pred = ols_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("OLS Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# Step 5: Save the model
ols_model_path = os.path.join("models", "currentOlsSolution.pkl")
os.makedirs("models", exist_ok=True)

with open(ols_model_path, "wb") as file:
    pickle.dump(ols_model, file)

print(f"OLS model saved to {ols_model_path}")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import os

# Dynamically construct file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file_path = os.path.join(script_dir, "../data/nepal-earthquake-severity-index-latest.csv")

# Output file paths
training_data_path = os.path.join(script_dir, "../data/training_data.csv")
test_data_path = os.path.join(script_dir, "../data/test_data.csv")
activation_data_path = os.path.join(script_dir, "../data/activation_data.csv")

# Step 1: Load dataset
data = pd.read_csv(input_file_path)
print("Dataset loaded successfully!")
print(data.head())

# Step 2: Clean the dataset
# Drop rows with missing values
data = data.dropna()

# Normalize numeric columns
scaler = MinMaxScaler()
numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Calculate z-scores and filter out outliers
z_scores = zscore(data[numeric_columns])
data["z_score"] = z_scores.mean(axis=1)
data = data[(data["z_score"].abs() < 3)]  # Keep rows with mean z-score < 3
data = data.drop("z_score", axis=1)  # Drop the temporary z-score column
print("Data cleaned and outliers removed!")

# Step 3: Split the dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Save the training and testing datasets
train_data.to_csv(training_data_path, index=False)
test_data.to_csv(test_data_path, index=False)

# Step 4: Create activation file
activation_data = test_data.sample(n=1, random_state=42)
activation_data.to_csv(activation_data_path, index=False)

print("Data preparation completed!")
print(f"Training data saved to {training_data_path}")
print(f"Testing data saved to {test_data_path}")
print(f"Activation data saved to {activation_data_path}")

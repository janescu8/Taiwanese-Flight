import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the synthetic dataset
data_path = "taiwanese_flight_data_2010_2019.csv"
synthetic_df = pd.read_csv(data_path)

# Filter data for years 2010 to 2018
synthetic_df["Date_of_Journey"] = pd.to_datetime(synthetic_df["Date_of_Journey"], format="%d/%m/%Y")
train_data = synthetic_df[synthetic_df["Date_of_Journey"].dt.year < 2019]

# Encode categorical features
label_encoders = {}
for column in ["Airline", "Source", "Destination", "Route", "Total_Stops", "Additional_Info"]:
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column])
    label_encoders[column] = le

# Select features and target variable
features = ["Airline", "Source", "Destination", "Route", "Total_Stops"]
target = "Price"

X = train_data[features]
y = train_data[target]

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

# Print evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the model to a .pkl file
model_path = "linear_regression_flight_fare.pkl"
joblib.dump({"model": model, "label_encoders": label_encoders}, model_path)

print(f"Model saved to {model_path}")

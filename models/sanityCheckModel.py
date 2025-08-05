import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# === Load dataset ===
df = pd.read_csv("../data/2022-23_to_2024-25_final.csv")

# === Define features and target ===
target = "total_points_next_gw"
features = df.columns.drop([target])

X = df[features]
y = df[target]

# === Quick train/test split (no shuffle for time-order) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# === Train model ===
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbosity=1
)

model.fit(X_train, y_train)

# === Predict & evaluate ===
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ RMSE on test set: {rmse:.3f}")
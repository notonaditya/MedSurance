import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("insurance.csv")

# Encode categorical variables (same as training)
label_encoders = {}
categorical_features = ["sex", "smoker", "region"]
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target variable
X = df.drop(columns=["charges"]).values.astype(float)
y = df["charges"].values.reshape(-1, 1).astype(float)

# Normalize numerical features (use same StandardScaler as training)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Convert to PyTorch tensors
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# Define the same model architecture as training
class InsuranceNN(nn.Module):
    def __init__(self, input_dim):
        super(InsuranceNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Load trained model
model = InsuranceNN(input_dim=X.shape[1]).to(device)
model.load_state_dict(torch.load("insurance_model_v2.pth"))
model.eval()

# Perform inference
with torch.no_grad():
    y_pred = model(X_tensor)

# Convert predictions back to original scale
y_actual = scaler_y.inverse_transform(y)
y_pred_actual = scaler_y.inverse_transform(y_pred.cpu().numpy())

# Compute evaluation metrics
mse = mean_squared_error(y_actual, y_pred_actual)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_actual, y_pred_actual)
r2 = r2_score(y_actual, y_pred_actual)

# Print results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

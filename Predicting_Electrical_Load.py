import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load datasets
ev_charging_reports = pd.read_csv('datasets/EV charging reports.csv')
traffic_reports = pd.read_csv('datasets/Local traffic distribution.csv')

# Merge datasets
ev_charging_traffic = ev_charging_reports.merge(
    traffic_reports, left_on='Start_plugin_hour', right_on='Date_from', how='inner')

# Drop unnecessary columns
drop_cols = ['session_ID', 'Garage_ID', 'User_ID', 'Shared_ID', 'Plugin_category',
             'Duration_category', 'Start_plugin', 'Start_plugin_hour', 'End_plugout',
             'End_plugout_hour', 'Date_from', 'Date_to']
ev_charging_traffic.drop(columns=drop_cols, inplace=True)

# Convert European decimal notation
ev_charging_traffic = ev_charging_traffic.replace({',': '.'}, regex=True)

# Convert columns to float
ev_charging_traffic = ev_charging_traffic.astype(float)

# Train-test split
X = ev_charging_traffic.drop(columns=['El_kWh'])
y = ev_charging_traffic['El_kWh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Linear regression baseline
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f'Linear Regression MSE: {test_mse}')

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define neural network
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 56),
    nn.ReLU(),
    nn.Linear(56, 26),
    nn.ReLU(),
    nn.Linear(26, 1)
)

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0007)

# Training loop
epochs = 3000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred_train = model(X_train_tensor)
    loss = loss_fn(y_pred_train, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Save model
torch.save(model.state_dict(), 'models/model.pth')

# Evaluate model
y_pred_test = model(X_test_tensor)
test_loss = loss_fn(y_pred_test, y_test_tensor).item()
print(f'Neural Network Test Loss: {test_loss}')

# Load longer-trained model and evaluate
long_trained_model = nn.Sequential(
    nn.Linear(X_train.shape[1], 56),
    nn.ReLU(),
    nn.Linear(56, 26),
    nn.ReLU(),
    nn.Linear(26, 1)
)
long_trained_model.load_state_dict(torch.load('models/model4500.pth'))
y_pred_long = long_trained_model(X_test_tensor)
test_loss_long = loss_fn(y_pred_long, y_test_tensor).item()
print(f'Longer-Trained Model Test Loss: {test_loss_long}')

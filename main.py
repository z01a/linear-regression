import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.linear_regression_gradient_descent import LinearRegressionGradientDescent

DATASET_PATH = 'datasets/house_prices_train.csv'

df = pd.read_csv(DATASET_PATH)

print("\nFirst 5 rows:\n", df.head())

print("\nLast 5 rows:\n", df.tail())

print("\nInfo:")
df.info()

print("\nDescribe:\n", df.describe())

features = ["Year_built", "Area", "Bath_no", "Bedroom_no"]

for feature in features:
    plt.figure()
    plt.scatter(df[feature], df["Price"])
    plt.xlabel(feature)
    plt.ylabel("Price")
    plt.title(f"Price vs {feature}")
    plt.tight_layout()
    plt.show()

X_raw = df[["Year_built", "Area"]]
y = df["Price"]

X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_raw, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)

X_train = pd.DataFrame(X_train_scaled, columns=X_train_raw.columns, index=X_train_raw.index)
X_val   = pd.DataFrame(X_val_scaled, columns=X_val_raw.columns, index=X_val_raw.index)

lrgd = LinearRegressionGradientDescent(learning_rate=0.01, n_iter=2000)
lrgd.fit(X_train, y_train)
y_pred_lrgd = lrgd.predict(X_val)

mse_lrgd  = mean_squared_error(y_val, y_pred_lrgd)
rmse_lrgd = np.sqrt(mse_lrgd)
r2_lrgd   = r2_score(y_val, y_pred_lrgd)

print("\n[LRGD] Intercept:", lrgd.coeff[0,0])
print("[LRGD] Coeffs:", dict(zip(X_train.columns.tolist(), lrgd.coeff[1:].reshape(-1))))
print("[LRGD] MSE:", mse_lrgd)
print("[LRGD] RMSE:", rmse_lrgd)
print("[LRGD] R^2:", r2_lrgd)

plt.figure()
plt.scatter(y_val, y_pred_lrgd, alpha=0.6)
plt.xlabel("Real price"); plt.ylabel("Predicted price")
plt.title("(LRGD) Real vs Predicted")
plt.tight_layout()
plt.show()

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)

mse_lr  = mean_squared_error(y_val, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr   = r2_score(y_val, y_pred_lr)

print("\n[LR] Intercept:", lr.intercept_)
print("[LR] Coeff:", dict(zip(X_train.columns.tolist(), lr.coef_)))
print("[LR] MSE:", mse_lr)
print("[LR] RMSE:", rmse_lr)
print("[LR] R^2:", r2_lr)

plt.figure()
plt.scatter(y_val, y_pred_lr, alpha=0.6)
plt.xlabel("Real price"); plt.ylabel("Predicted price")
plt.title("(LR) Real vs Predicted")
plt.tight_layout()
plt.show()

spots = 200
area_min, area_max = X_raw["Area"].min(), X_raw["Area"].max()
area_grid = np.linspace(area_min, area_max, num=spots)
year_fixed = X_raw["Year_built"].mean()

estates = pd.DataFrame({
    "Year_built": np.full(spots, year_fixed),
    "Area": area_grid
})

estates_scaled_np = scaler.transform(estates)
estates_scaled = pd.DataFrame(estates_scaled_np, columns=X_train.columns)

y_line_lrgd = lrgd.predict(estates_scaled)
y_line_lr   = lr.predict(estates_scaled)

plt.figure(figsize=(9, 6))
plt.scatter(X_val_raw["Area"], y_val, alpha=0.35, label="Samples", s=25)
plt.plot(area_grid, y_line_lrgd, lw=3, label="LRGD Model")
plt.plot(area_grid, y_line_lr, lw=2, label="LR Model")
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Price vs Area")
plt.legend()
plt.tight_layout()
plt.show()
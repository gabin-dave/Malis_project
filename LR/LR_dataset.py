import pandas as pd
import numpy as np

# --- Create 500 samples ---
np.random.seed(42)
n = 500

# Independent variables
size = np.random.normal(120, 30, n)                  # square meters
bedrooms = np.random.randint(1, 6, n)                # 1–5 bedrooms
distance_to_city = np.random.uniform(1, 20, n)       # km
house_age = np.random.randint(1, 50, n)              # years
income_neighborhood = np.random.normal(60_000, 10_000, n)  # yearly income

# Target variable (house price)
noise = np.random.normal(0, 20_000, n)
house_price = (
    30000 + size*1500 + bedrooms*10000
    - distance_to_city*3000 - house_age*500
    + (income_neighborhood/10) + noise
)

# Combine into a dataframe
data = pd.DataFrame({
    "size": size,
    "bedrooms": bedrooms,
    "distance_to_city": distance_to_city,
    "house_age": house_age,
    "income_neighborhood": income_neighborhood,
    "house_price": house_price
})

# Save to CSV
data.to_csv("house_price_dataset.csv", index=False)
print(data.head())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = data.drop("house_price", axis=1)
y = data["house_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")

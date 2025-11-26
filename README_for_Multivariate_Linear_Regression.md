# Rental Prices Dataset for Multivariate Linear Regression

## 1. Project Overview  
The dataset models rental prices for apartments in an urban environment, using multiple features related to the property, the building, and the neighborhood.  
It aims to simulate a real-world scenario where rental agencies or property analysts try to predict monthly rent based on observable characteristics.

---

## 2. Dataset Description
The dataset contains **1,000 observations**, each representing an apartment.  
The following variables are included:

### **Apartment Features**
- `size_sqm` — Surface area in square meters  
- `num_bedrooms` — Number of bedrooms (1 to 5)  
- `is_furnished` — Binary indicator for furnishing

### **Building Characteristics**
- `floor` — Floor number (0 = ground floor)  
- `has_elevator` — Elevator presence (0/1)  
- `has_balcony` — Balcony presence (0/1)  
- `building_age_years` — Age of the building in years

### **Location & Neighborhood**
- `distance_to_center_km` — Distance from the city center  
- `neighborhood_income_index` — Socioeconomic score of the area (0 to 1)  
- `noise_level_index` — Ambient noise level (0 to 1)

### **Target Variable**
- `monthly_rent_eur` — Monthly rent in euros (value to predict)

These features were generated using statistical distributions and dependencies to reflect realistic housing patterns (e.g., higher floors more likely to have balconies, larger apartments in wealthier neighborhoods, etc.).

---

## 3. File Structure
The repository contains:

- **`dataset_generation.ipynb`** — Jupyter Notebook used to generate the dataset  
- **`rental_prices.csv`** — Final dataset in CSV format

---

## 4. Dataset Generation Method
The dataset was generated programmatically using Python (`numpy` and `pandas`).  
A linear model defined the true relationship between property characteristics and rent, with added Gaussian noise to simulate real-world variability.

Key modeling choices include:
- Correlation between neighborhood income and apartment size  
- Floor-dependent probability of having a balcony  
- Surface-dependent probability of number of bedrooms  
- Realistic ranges for age, distance to center, and noise levels

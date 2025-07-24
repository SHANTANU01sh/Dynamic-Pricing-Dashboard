# dynamic_pricing_model.py

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load datasets
print("✅ Loading datasets...")

amazon_df = pd.read_csv(r"C:\Users\KIIT\Desktop\data_engineer_challenge\data\Amazon Sale Report.csv", low_memory=False)
cloud_df = pd.read_csv(r"C:\Users\KIIT\Desktop\data_engineer_challenge\data\Cloud Warehouse Compersion Chart.csv")
expense_df = pd.read_csv(r"C:\Users\KIIT\Desktop\data_engineer_challenge\data\Expense IIGF.csv")
intl_df = pd.read_csv(r"C:\Users\KIIT\Desktop\data_engineer_challenge\data\International sale Report.csv")
may_df = pd.read_csv(r"C:\Users\KIIT\Desktop\data_engineer_challenge\data\May-2022.csv")
pl_df = pd.read_csv(r"C:\Users\KIIT\Desktop\data_engineer_challenge\data\P  L March 2021.csv")
sale_df = pd.read_csv(r"C:\Users\KIIT\Desktop\data_engineer_challenge\data\Sale Report.csv")

# Step 2: Preprocessing
print("🔧 Preprocessing...")

def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

amazon_df = clean_columns(amazon_df)
cloud_df = clean_columns(cloud_df)
expense_df = clean_columns(expense_df)
intl_df = clean_columns(intl_df)
may_df = clean_columns(may_df)
pl_df = clean_columns(pl_df)
sale_df = clean_columns(sale_df)

# Check basic info
print("Amazon DF shape:", amazon_df.shape)
print("Sale DF shape:", sale_df.shape)
print("Sale DF sample:")
print(sale_df.head())

# Step 3: Generate synthetic required columns
print("⚙️ Generating synthetic data...")

# Rename sku_code to sku for merging later
sale_df.rename(columns={'sku_code': 'sku'}, inplace=True)

# Create synthetic quantity and unit_price columns
np.random.seed(42)
sale_df['quantity'] = np.random.randint(1, 10, size=len(sale_df))
sale_df['unit_price'] = np.random.uniform(50, 200, size=len(sale_df)).round(2)

sale_df['total_amount'] = sale_df['quantity'] * sale_df['unit_price']

# Aggregate to get SKU level features
agg_df = sale_df.groupby('sku').agg({
    'quantity': 'sum',
    'unit_price': 'mean',
    'total_amount': 'sum'
}).reset_index()

agg_df.rename(columns={
    'quantity': 'total_quantity',
    'unit_price': 'avg_unit_price',
    'total_amount': 'total_sales'
}, inplace=True)

# Merge with amazon_df if sku column matches
if 'sku' in amazon_df.columns:
    merged_df = pd.merge(agg_df, amazon_df, on='sku', how='left')
else:
    merged_df = agg_df.copy()

# Fill missing numerical values with mean
for col in merged_df.select_dtypes(include=np.number).columns:
    merged_df[col].fillna(merged_df[col].mean(), inplace=True)

# Show final dataset
print("Merged dataset sample:")
print(merged_df.head())

# Step 4: Model training
print("🤖 Training models...")

features = ['total_quantity', 'avg_unit_price']
target = 'total_sales'

X = merged_df[features]
y = merged_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

results = {}

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    cv_rmse = -cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error').mean()
    print(f"{name} RMSE: {rmse:.2f}, CV RMSE: {cv_rmse:.2f}")
    results[name] = (model, rmse, cv_rmse)

    # Save model to file
    model_filename = f"models/{name}.pkl"
    joblib.dump(model, model_filename)
    print(f"💾 Saved {name} model to {model_filename}")

# Pick best model
best_model_name = min(results, key=lambda k: results[k][2])
best_model = results[best_model_name][0]
print(f"\n✅ Best model: {best_model_name}")

merged_df['suggested_price'] = best_model.predict(merged_df[features])

# Save results
os.makedirs("output", exist_ok=True)
merged_df.to_parquet('output/suggested_prices.parquet', index=False)

print("🎉 Done! Predictions saved to output/suggested_prices.parquet")

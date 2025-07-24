# pipeline.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
import mlflow
import mlflow.sklearn

def load_and_clean_data(data_dir='data'):
    print("✅ Loading datasets...")
    amazon_df = pd.read_csv(r"C:\Users\KIIT\Desktop\data_engineer_challenge\data\Amazon Sale Report.csv", low_memory=False)
    sale_df = pd.read_csv(r"C:\Users\KIIT\Desktop\data_engineer_challenge\data\Sale Report.csv")

    def clean(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        return df

    amazon_df = clean(amazon_df)
    sale_df = clean(sale_df)

    return amazon_df, sale_df

def generate_synthetic(sale_df):
    print("⚙️ Generating synthetic data...")
    sale_df.rename(columns={'sku_code': 'sku'}, inplace=True)
    np.random.seed(42)
    sale_df['quantity'] = np.random.randint(1, 10, size=len(sale_df))
    sale_df['unit_price'] = np.random.uniform(50, 200, size=len(sale_df)).round(2)
    sale_df['total_amount'] = sale_df['quantity'] * sale_df['unit_price']
    return sale_df

def feature_engineering(sale_df, amazon_df):
    print("🔧 Feature engineering...")
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

    if 'sku' in amazon_df.columns:
        merged = pd.merge(agg_df, amazon_df, on='sku', how='left')
    else:
        merged = agg_df

    for col in merged.select_dtypes(include=np.number).columns:
        merged[col] = merged[col].fillna(merged[col].mean())

    print("✅ Merged sample:")
    print(merged.head())
    return merged

def train_and_evaluate(X, y):
    print("🤖 Training models with MLflow logging...")
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs("models", exist_ok=True)

    mlflow.set_experiment("Dynamic Pricing Experiment")

    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        with mlflow.start_run(run_name=name):
            # Fit & predict
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            cv_rmse = -cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error').mean()
            print(f"{name} RMSE: {rmse:.2f}, CV RMSE: {cv_rmse:.2f}")

            # Log parameters & metrics
            mlflow.log_param("model_name", name)
            if hasattr(model, 'n_estimators'):
                mlflow.log_param("n_estimators", getattr(model, 'n_estimators', None))
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("cv_rmse", cv_rmse)

            # Save locally
            local_model_path = f"models/{name}.pkl"
            joblib.dump(model, local_model_path)
            print(f"💾 Saved to {local_model_path}")

            # Log model file as artifact
            mlflow.log_artifact(local_model_path)

            # Optionally log as MLflow model for serving:
            mlflow.sklearn.log_model(model, artifact_path="model")

            results[name] = (model, rmse, cv_rmse)

    # Pick best model by lowest CV RMSE
    best_model_name = min(results, key=lambda k: results[k][2])
    best_model = results[best_model_name][0]
    print(f"\n✅ Best model: {best_model_name}")
    return best_model

def save_predictions(df, model, features, out_path='output/suggested_prices.parquet'):
    df['suggested_price'] = model.predict(df[features])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"🎉 Predictions saved to {out_path}")

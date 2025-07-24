# flow.py
from prefect import flow
from pipeline import load_and_clean_data, generate_synthetic, feature_engineering, train_and_evaluate, save_predictions

@flow
def dynamic_pricing_flow():
    amazon_df, sale_df = load_and_clean_data()
    sale_df = generate_synthetic(sale_df)
    merged_df = feature_engineering(sale_df, amazon_df)
    X = merged_df[['total_quantity', 'avg_unit_price']]
    y = merged_df['total_sales']
    best_model = train_and_evaluate(X, y)
    save_predictions(merged_df, best_model, ['total_quantity', 'avg_unit_price'])

if __name__ == "__main__":
    dynamic_pricing_flow()

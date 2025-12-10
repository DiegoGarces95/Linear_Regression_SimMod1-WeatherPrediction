"""
linear_regression_model.py
Script for training, evaluation, and visualization of a multiple linear regression model.
Includes detailed comments for learning.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def train_and_evaluate(data, test_size=0.2, random_state=42):
    """
    Trains a multiple linear regression model to predict mean temperature (TG).
    Splits data into training and testing sets, trains the model, and evaluates its performance.
    Displays metrics and a plot of actual vs. predicted results.
    """
    # 1. Select predictor variables and target
    X = data.drop(columns=['DATE', 'TG'])
    y = data['TG']

    # 2. Split into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Make predictions
    y_pred = model.predict(X_test)

    # 5. Evaluate the model with standard metrics
    mae = mean_absolute_error(y_test, y_pred)
    # Calcluate RMSE manually for compatibility
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")

    # 6. Result visualization
    plt.figure(figsize=(10,4))
    plt.plot(y_test.values, label='Actual', marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred, label='Predicted', marker='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Actual vs Predicted Temperature (Test Set)')
    plt.xlabel('Sample Index')
    plt.ylabel('Temperature (°C)')
    plt.tight_layout()
    plt.show()

    return model

def linear_regression_experiments(train_df, test_df, figures_dir="figures"):
    """
    Performs two experiments:
    1. Simple Linear Regression: TG_lag1 -> TG
    2. Multiple Linear Regression: TN, TX, TG -> TG
    Saves results and plots.
    """
    os.makedirs(figures_dir, exist_ok=True)
    # --- Simple Linear Regression ---
    train_df["TG_lag1"] = train_df["TG"].shift(1)
    test_df["TG_lag1"] = test_df["TG"].shift(1)
    train_simple = train_df.dropna()
    test_simple = test_df.dropna()
    X_train_simple = train_simple[["TG_lag1"]]
    y_train_simple = train_simple["TG"]
    X_test_simple = test_simple[["TG_lag1"]]
    y_test_simple = test_simple["TG"]
    model_simple = LinearRegression()
    model_simple.fit(X_train_simple, y_train_simple)
    pred_simple = model_simple.predict(X_test_simple)
    mae_simple = mean_absolute_error(y_test_simple, pred_simple)
    rmse_simple = mean_squared_error(y_test_simple, pred_simple) ** 0.5
    r2_simple = r2_score(y_test_simple, pred_simple)
    results_simple = pd.DataFrame({
        "date": test_simple["DATE"],
        "actual_TG": y_test_simple,
        "predicted_TG": pred_simple
    })
    results_simple.to_csv("linear_regression_predictions_simple.csv", index=False)
    plt.figure(figsize=(10,4))
    plt.plot(y_test_simple.values, label='Actual', marker='o', linestyle='-', alpha=0.7)
    plt.plot(pred_simple, label='Predicted', marker='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Simple Linear Regression: Actual vs Predicted TG')
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Temperature (°C)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'tg_simple_real_vs_predicho.png'))
    plt.close()
    print("--- Simple Linear Regression ---")
    print(f"MAE: {mae_simple:.3f}")
    print(f"RMSE: {rmse_simple:.3f}")
    print(f"R²: {r2_simple:.3f}")
    print("Results saved in linear_regression_predictions_simple.csv and figures/tg_simple_real_vs_predicho.png\n")
    # --- Multiple Linear Regression ---
    feature_cols = ["TN", "TX", "TG"]
    X_train_multi = train_df[feature_cols]
    y_train_multi = train_df["TG"]
    X_test_multi = test_df[feature_cols]
    y_test_multi = test_df["TG"]
    model_multi = LinearRegression()
    model_multi.fit(X_train_multi, y_train_multi)
    pred_multi = model_multi.predict(X_test_multi)
    mae_multi = mean_absolute_error(y_test_multi, pred_multi)
    rmse_multi = mean_squared_error(y_test_multi, pred_multi) ** 0.5
    r2_multi = r2_score(y_test_multi, pred_multi)
    results_multi = pd.DataFrame({
        "date": test_df["DATE"],
        "actual_TG": y_test_multi,
        "predicted_TG": pred_multi
    })
    results_multi.to_csv("linear_regression_predictions_multi.csv", index=False)
    plt.figure(figsize=(10,4))
    plt.plot(y_test_multi.values, label='Actual', marker='o', linestyle='-', alpha=0.7)
    plt.plot(pred_multi, label='Predicted', marker='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Multiple Linear Regression: Actual vs Predicted TG')
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Temperature (°C)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'tg_multi_real_vs_predicho.png'))
    plt.close()
    print("--- Multiple Linear Regression ---")
    print(f"MAE: {mae_multi:.3f}")
    print(f"RMSE: {rmse_multi:.3f}")
    print(f"R²: {r2_multi:.3f}")
    print("Results saved in linear_regression_predictions_multi.csv and figures/tg_multi_real_vs_predicho.png\n")

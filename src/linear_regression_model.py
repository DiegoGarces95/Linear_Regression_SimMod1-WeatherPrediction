"""
linear_regression_model.py
Script para entrenamiento, evaluación y visualización de un modelo de regresión lineal múltiple.
Incluye comentarios detallados para aprendizaje.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def train_and_evaluate(data, test_size=0.2, random_state=42):
    """
    Entrena un modelo de regresión lineal múltiple para predecir la temperatura media (TG).
    Divide los datos en entrenamiento y prueba, entrena el modelo y evalúa su desempeño.
    Muestra métricas y una gráfica de resultados reales vs. predichos.
    """
    # 1. Seleccionar variables predictoras y objetivo
    X = data.drop(columns=['DATE', 'TG'])
    y = data['TG']

    # 2. Dividir en conjunto de entrenamiento y prueba
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Realizar predicciones
    y_pred = model.predict(X_test)

    # 5. Evaluar el modelo con métricas estándar
    mae = mean_absolute_error(y_test, y_pred)
    # Para compatibilidad con versiones antiguas de scikit-learn, calcular RMSE manualmente
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")

    # 6. Visualización de resultados
    plt.figure(figsize=(10,4))
    plt.plot(y_test.values, label='Real', marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred, label='Predicho', marker='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Temperatura real vs predicha (conjunto de prueba)')
    plt.xlabel('Índice de muestra')
    plt.ylabel('Temperatura (°C)')
    plt.tight_layout()
    plt.show()

    return model

def linear_regression_experiments(train_df, test_df, figures_dir="figures"):
    """
    Realiza dos experimentos:
    1. Regresión lineal simple: TG_lag1 -> TG
    2. Regresión lineal múltiple: TN, TX, TG -> TG
    Guarda resultados y gráficas.
    """
    os.makedirs(figures_dir, exist_ok=True)
    # --- Regresión lineal simple ---
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
    plt.plot(y_test_simple.values, label='Real', marker='o', linestyle='-', alpha=0.7)
    plt.plot(pred_simple, label='Predicho', marker='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Regresión lineal simple: TG real vs predicha')
    plt.xlabel('Índice de muestra')
    plt.ylabel('Temperatura media (°C)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'tg_simple_real_vs_predicho.png'))
    plt.close()
    print("--- Regresión lineal simple ---")
    print(f"MAE: {mae_simple:.3f}")
    print(f"RMSE: {rmse_simple:.3f}")
    print(f"R²: {r2_simple:.3f}")
    print("Resultados guardados en linear_regression_predictions_simple.csv y figures/tg_simple_real_vs_predicho.png\n")
    # --- Regresión lineal múltiple ---
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
    plt.plot(y_test_multi.values, label='Real', marker='o', linestyle='-', alpha=0.7)
    plt.plot(pred_multi, label='Predicho', marker='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Regresión lineal múltiple: TG real vs predicha')
    plt.xlabel('Índice de muestra')
    plt.ylabel('Temperatura media (°C)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'tg_multi_real_vs_predicho.png'))
    plt.close()
    print("--- Regresión lineal múltiple ---")
    print(f"MAE: {mae_multi:.3f}")
    print(f"RMSE: {rmse_multi:.3f}")
    print(f"R²: {r2_multi:.3f}")
    print("Resultados guardados en linear_regression_predictions_multi.csv y figures/tg_multi_real_vs_predicho.png\n")

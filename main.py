"""
main.py

Script principal para ejecutar el flujo completo de predicción de temperatura con regresión lineal múltiple.
Cada paso está documentado para aprendizaje.
"""
from src.data_loader import load_weather_data, create_lagged_features
from src.linear_regression_model import train_and_evaluate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __name__ == "__main__":
    # 1. Cargar y limpiar los datos
    data = load_weather_data("data")
    print("\nDatos cargados y limpios (primeras filas):")
    print(data.head())

    print("\n--- Estadísticas Descriptivas ---")
    print(data.describe())

    # 2. Preparar datos con rezagos (lags)
    lagged_data = create_lagged_features(data, n_lags=3)
    print("\nDatos con rezagos (primeras filas):")
    print(lagged_data.head())

    # 3. Entrenamiento y evaluación del modelo
    model = train_and_evaluate(lagged_data)

    # 4. Dashboard de resultados

    X = lagged_data.drop(columns=["DATE", "TG"])
    y = lagged_data["TG"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    print("\n--- Dashboard de Resultados ---")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred)**0.5:.3f}")
    print(f"R²: {r2_score(y_test, y_pred):.3f}")

    plt.figure(figsize=(10,4))
    plt.plot(y_test.values, label='Real', marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred, label='Predicho', marker='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Temperatura real vs predicha (conjunto de prueba)')
    plt.xlabel('Índice de muestra')
    plt.ylabel('Temperatura (°C)')
    plt.tight_layout()
    plt.savefig('figures/real_vs_predicho.png')
    plt.show()

    # 5. Conclusiones y sugerencias
    print("\nConclusión: El modelo de regresión lineal con rezagos permite predecir la temperatura media diaria usando los valores de días anteriores. Puedes experimentar con el número de rezagos para mejorar el desempeño.")

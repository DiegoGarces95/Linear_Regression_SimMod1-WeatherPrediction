
"""
linear_regression_model.py
Script para entrenamiento, evaluación y visualización de un modelo de regresión lineal múltiple.
Incluye comentarios detallados para aprendizaje.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

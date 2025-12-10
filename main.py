"""
main.py

Script principal para ejecutar el flujo completo de predicción de temperatura con regresión lineal múltiple.
Cada paso está documentado para aprendizaje.
"""
from src.data_loader import load_weather_data
from src.data_exploration import describe_data, plot_histograms, plot_correlation_heatmap
from src.data_preparation import select_features_and_target, create_sliding_windows, normalize_data
from src.linear_regression_model import train_and_evaluate

if __name__ == "__main__":
    # 1. Cargar y limpiar los datos
    data = load_weather_data("data")
    print("\nDatos cargados y limpios (primeras filas):")
    print(data.head())

    # 2. Análisis exploratorio
    describe_data(data)
    plot_histograms(data)
    plot_correlation_heatmap(data)

    # 3. Preparación de datos (puedes personalizar aquí)
    # Ejemplo simple: usar todas las variables y predecir TG
    X, y = select_features_and_target(data)
    # Si quieres usar ventanas temporales, descomenta:
    # X, y = create_sliding_windows(data, input_days=3, pred_days=1)

    # 4. Entrenamiento y evaluación del modelo
    # Por simplicidad, usamos el DataFrame original (train_test_split está en el modelo)
    model = train_and_evaluate(data)

    # 5. Conclusiones y sugerencias
    print("\nConclusión: El modelo de regresión lineal múltiple permite predecir la temperatura media diaria usando variables meteorológicas históricas.\nPuedes experimentar con ventanas temporales, normalización y selección de variables para mejorar el desempeño.")

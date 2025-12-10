"""
main.py

Script principal para ejecutar el flujo completo de predicción de temperatura con regresión lineal múltiple.
Cada paso está documentado para aprendizaje.
"""
import os
from src.data_loader import load_datasets
from src.linear_regression_model import linear_regression_experiments

if __name__ == "__main__":
    # Crear directorio para figuras si no existe
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Cargar los datasets
    train_df, val_df, test_df = load_datasets("data")

    # Ejecutar los experimentos de regresión lineal
    linear_regression_experiments(train_df, test_df, figures_dir=figures_dir)

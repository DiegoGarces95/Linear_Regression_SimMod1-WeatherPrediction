"""
data_exploration.py

Script para análisis exploratorio de los datos meteorológicos.
Incluye estadísticas descriptivas y visualización de correlaciones.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def describe_data(data):
    """
    Muestra estadísticas descriptivas básicas de las variables.
    """
    print("\n--- Estadísticas Descriptivas ---")
    print(data.describe())

def plot_histograms(data):
    """
    Genera histogramas para cada variable numérica.
    """
    # Excluir columnas no numéricas y columnas que parecen numéricas pero son fechas codificadas como string/object
    numeric_data = data.copy()
    # Eliminar la columna 'DATE' si existe
    if 'DATE' in numeric_data.columns:
        numeric_data = numeric_data.drop(columns=['DATE'])
    # Seleccionar solo columnas de tipo float o int
    numeric_data = numeric_data.select_dtypes(include=['float64', 'int64'])
    if numeric_data.shape[1] == 0:
        print("No hay columnas numéricas para graficar.")
        return
    numeric_data.hist(bins=30, figsize=(14,8))
    plt.suptitle('Distribución de variables meteorológicas')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_correlation_heatmap(data):
    """
    Muestra un mapa de calor de correlaciones entre variables.
    """
    plt.figure(figsize=(10,8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de correlación entre variables')
    plt.show()

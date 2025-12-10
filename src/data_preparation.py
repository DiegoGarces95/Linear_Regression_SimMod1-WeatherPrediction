"""
data_preparation.py

Script para preparar los datos para el modelo de regresión lineal múltiple.
Incluye selección de variables, creación de ventanas temporales y normalización.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

def select_features_and_target(data, target_col='TG'):
    """
    Separa las variables predictoras (X) y la variable objetivo (y).
    Por defecto, la variable objetivo es 'TG'.
    """
    X = data.drop(columns=['DATE', target_col])
    y = data[target_col]
    return X, y

def create_sliding_windows(data, input_days=1, pred_days=1, target_col='TG'):
    """
    Crea ventanas deslizantes para predicción a varios días.
    input_days: número de días de entrada (histórico)
    pred_days: número de días a predecir (horizonte)
    Devuelve X (matriz de entrada) y y (matriz de salida)
    """
    X_list, y_list = [], []
    for i in range(len(data) - input_days - pred_days + 1):
        X_window = data.iloc[i:i+input_days].drop(columns=['DATE', target_col]).values.flatten()
        y_window = data.iloc[i+input_days:i+input_days+pred_days][target_col].values
        X_list.append(X_window)
        y_list.append(y_window)
    return pd.DataFrame(X_list), pd.DataFrame(y_list)

def normalize_data(X_train, X_test):
    """
    Normaliza los datos de entrada usando StandardScaler (media 0, varianza 1).
    Ajusta el scaler solo con los datos de entrenamiento.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

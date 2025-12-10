
import pandas as pd
import os

def load_weather_data(data_dir):
    """
    Lee y combina los archivos de datos meteorológicos relevantes.
    Realiza limpieza básica: elimina valores faltantes y unifica por fecha.
    Devuelve un DataFrame listo para análisis y modelado.
    """
    # Cargar solo la temperatura media diaria desde el archivo CSV
    path = os.path.join(data_dir, 'TG_SOUID121044_1.csv')
    df = pd.read_csv(path, sep=';', skipinitialspace=True)
    # Renombrar columnas por claridad
    df = df.rename(columns={"DATE": "DATE", "TG": "TG"})
    # Convertir TG a numérico y marcar valores faltantes (-9999)
    df['TG'] = pd.to_numeric(df['TG'], errors='coerce')
    df = df[df['TG'] != -9999]
    # Convertir TG de décimas a grados Celsius
    df['TG'] = df['TG'] / 10.0
    # Mantener solo fecha y TG
    df = df[['DATE', 'TG']]
    # Resetear el índice y devolver el DataFrame limpio
    df = df.reset_index(drop=True)
    return df

def create_lagged_features(df, n_lags=3):
    """
    Crea un DataFrame con variables de rezago (lags) de TG para predicción.
    Cada fila contiene TG(t), TG(t-1), TG(t-2), ..., TG(t-n_lags).
    Elimina filas con valores faltantes generados por el desplazamiento.
    """
    df_lagged = df.copy()
    for lag in range(1, n_lags+1):
        df_lagged[f'TG_lag_{lag}'] = df_lagged['TG'].shift(lag)
    # Eliminar filas con valores faltantes
    df_lagged = df_lagged.dropna().reset_index(drop=True)
    # Mantener solo las columnas relevantes
    cols = ['DATE', 'TG'] + [f'TG_lag_{lag}' for lag in range(1, n_lags+1)]
    return df_lagged[cols]

  
"""
data_loader.py

Script para cargar y limpiar los datos meteorológicos ECA&D para regresión lineal múltiple.
Incluye comentarios detallados para aprendizaje.
"""
import pandas as pd
import os

def load_weather_data(data_dir):
    """
    Lee y combina los archivos de datos meteorológicos relevantes.
    Realiza limpieza básica: elimina valores faltantes y unifica por fecha.
    Devuelve un DataFrame listo para análisis y modelado.
    """
    # Diccionario con los nombres de archivo de cada variable relevante
    files = {
        'TG': 'TG_SOUID121044.txt',   # Temperatura media diaria (objetivo)
        'TN': 'TN_SOUID121045.txt',   # Temperatura mínima diaria
        'TX': 'TX_SOUID121046.txt',   # Temperatura máxima diaria
        'RR': 'RR_SOUID121042.txt',   # Precipitación
        'SS': 'SS_SOUID121040.txt',   # Horas de sol
        'HU': 'HU_SOUID121047.txt',   # Humedad
        'FG': 'FG_SOUID121048.txt',   # Velocidad del viento
        'FX': 'FX_SOUID121049.txt',   # Ráfaga máxima de viento
        'CC': 'CC_SOUID121039.txt',   # Nubosidad
        'SD': 'SD_SOUID121043.txt',   # Profundidad de nieve
    }

    dfs = []  # Lista para almacenar los DataFrames de cada variable
    for var, fname in files.items():
        path = os.path.join(data_dir, fname)
        # Leer el archivo, saltando encabezados y usando nombres de columnas personalizados
        df = pd.read_csv(path, comment='#', skiprows=11, names=['SOUID','DATE',var,'Q_'+var])
        df = df[['DATE', var]]  # Nos quedamos solo con la fecha y la variable
        # Convertir a numérico y marcar valores faltantes (-9999)
        df[var] = pd.to_numeric(df[var], errors='coerce')
        df[var] = df[var].replace(-9999, pd.NA)
        dfs.append(df)

    # Unir todos los DataFrames por la columna 'DATE' (intersección de fechas)
    from functools import reduce
    data = reduce(lambda left, right: pd.merge(left, right, on='DATE', how='inner'), dfs)

    # Eliminar filas con cualquier valor faltante
    data = data.dropna()

    # Conversión de unidades a valores reales
    # TG, TN, TX, RR, SS, FG, FX están en décimas (dividir por 10)
    for var in ['TG','TN','TX','RR','SS','FG','FX']:
        data[var] = data[var] / 10.0

    # HU (humedad), CC (nubosidad), SD (nieve) ya están en unidades estándar

    # Resetear el índice y devolver el DataFrame limpio
    data = data.reset_index(drop=True)
    return data

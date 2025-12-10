# Predicción de Temperatura con Regresión Lineal Múltiple

Este proyecto es una guía de aprendizaje para predecir la temperatura media diaria (TG) utilizando regresión lineal múltiple y datos históricos meteorológicos del European Climate Assessment & Dataset (ECA&D).

## Estructura del Proyecto

- `src/data_loader.py`: Carga y limpieza de los datos meteorológicos relevantes.
- `src/linear_regression_model.py`: Entrenamiento, evaluación y visualización del modelo de regresión lineal.
- `main.py`: Script principal para ejecutar todo el flujo.
- `data/`: Carpeta con los archivos de datos originales de ECA&D.
- `figures/`: Carpeta sugerida para guardar gráficos generados.
- `notebooks/`: Para experimentos y análisis exploratorio.

## Variables Utilizadas

- **TG**: Temperatura media diaria (objetivo)
- **TN**: Temperatura mínima diaria
- **TX**: Temperatura máxima diaria
- **RR**: Precipitación
- **SS**: Horas de sol
- **HU**: Humedad
- **FG**: Velocidad del viento
- **FX**: Ráfaga máxima de viento
- **CC**: Nubosidad
- **SD**: Profundidad de nieve

## Requisitos

- Python 3.7+
- pandas
- scikit-learn
- matplotlib

Instala dependencias con:
```bash
pip install pandas scikit-learn matplotlib
```


## Resultados y Experimentos

### 1. Carga y limpieza de datos

- Se utilizó únicamente la temperatura media diaria (TG) del archivo `TG_SOUID121044_1.csv`.
- Se eliminaron los valores faltantes (-9999).
- Total de registros válidos: **24,867**.
- Ejemplo de los primeros datos:

|   DATE   |  TG  |
|----------|------|
|19570901  | 14.4 |
|19570902  | 13.2 |
|19570903  | 12.3 |
|19570904  | 12.6 |
|19570905  | 16.4 |

### 2. Plan de experimentos

- **Regresión lineal simple:** Usar solo TG para predecir la temperatura media.
- **Regresión lineal múltiple:** Usar TG, TN y TX para comparar resultados.
- Se incluirán gráficas de ajuste y métricas de desempeño.

### 3. Regresión lineal con rezagos (lags)

Se entrenó un modelo de regresión lineal para predecir la temperatura media diaria (TG) usando los valores de TG de los 3 días anteriores como variables predictoras.

**Resultados:**
- MAE: 1.779
- RMSE: 2.267
- R²: 0.892

Esto muestra una mejora significativa respecto al modelo simple, ya que el modelo ahora captura la dependencia temporal de la serie.

**Conclusión:**
Utilizar los valores de días anteriores permite obtener predicciones mucho más precisas y coherentes para la temperatura media diaria.

---

## Ejecución Paso a Paso

1. Coloca los archivos de datos en la carpeta `data/`.

2. **Exploración y limpieza de datos:**
	- El script `src/data_loader.py` carga y limpia los datos meteorológicos relevantes, eliminando valores faltantes y unificando por fecha.

3. **Análisis exploratorio:**
	- Usa `src/data_exploration.py` para mostrar estadísticas descriptivas, histogramas y el mapa de correlación entre variables.
	- Ejemplo de uso:
	  ```python
	  from src.data_loader import load_weather_data
	  from src.data_exploration import describe_data, plot_histograms, plot_correlation_heatmap
	  data = load_weather_data("data")
	  describe_data(data)
	  plot_histograms(data)
	  plot_correlation_heatmap(data)
	  ```


4. **Preparación de datos:**
	 - Usa `src/data_preparation.py` para seleccionar variables, crear ventanas temporales y normalizar los datos.
	 - Ejemplo de uso:
		 ```python
		 from src.data_preparation import select_features_and_target, create_sliding_windows, normalize_data
		 # Selección simple
		 X, y = select_features_and_target(data)
		 # O para ventanas temporales (ejemplo: 3 días de entrada, 2 de predicción)
		 X_win, y_win = create_sliding_windows(data, input_days=3, pred_days=2)
		 # Normalización
		 X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
		 ```


5. **Modelado y evaluación:**
	 - Usa `src/linear_regression_model.py` para entrenar y evaluar el modelo de regresión lineal múltiple.
	 - El script divide los datos en entrenamiento y prueba, entrena el modelo y muestra métricas (MAE, RMSE, R²) y una gráfica de resultados reales vs. predichos.
	 - Ejemplo de uso:
		 ```python
		 from src.linear_regression_model import train_and_evaluate
		 model = train_and_evaluate(data)
		 ```

	 - O ejecuta el flujo completo (carga, exploración, preparación, modelado y visualización) con:
		 ```bash
		 python main.py
		 ```
	 - El script principal está documentado paso a paso para aprendizaje y puedes personalizar cada etapa según tus necesidades.

### 4. Visualización de resultados

La siguiente gráfica muestra la comparación entre la temperatura real y la predicha por el modelo de regresión lineal (usando los valores de TG de los 3 días anteriores):

![Temperatura real vs predicha](figures/real_vs_predicho.png)

Puedes encontrar la imagen generada en la carpeta `figures/` después de ejecutar `main.py`.

---

## Notas
- Los datos con valores faltantes o erróneos (-9999) se eliminan automáticamente.
- Las variables de temperatura, precipitación, viento y sol se convierten a sus unidades estándar (°C, mm, m/s, h).
- Puedes modificar los scripts en `src/` para experimentar con nuevas variables, ventanas temporales o análisis adicionales.

## Licencia

Este proyecto utiliza datos de ECA&D. Consulta los términos de uso en https://www.ecad.eu/dailydata/index.php

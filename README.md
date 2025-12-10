# Temperature Prediction with Linear Regression

This project is a learning guide for predicting daily mean temperature (TG) using multiple linear regression and historical weather data from the European Climate Assessment & Dataset (ECA&D).

## Project Structure

- `src/data_loader.py`: Loading and cleaning of relevant weather data.
- `src/linear_regression_model.py`: Training, evaluation, and visualization of the linear regression model.
- `main.py`: Main script to execute the entire workflow.
- `data/`: Folder containing the original data files.
- `figures/`: Folder for generated plots.
- `notebooks/`: For experiments and exploratory analysis.

## Variables Used

- **TG**: Mean daily temperature (Target)
- **TN**: Minimum daily temperature
- **TX**: Maximum daily temperature

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- matplotlib

Install dependencies with:
```bash
pip install pandas scikit-learn matplotlib
```

## Methodology & Experiments

The project implements two experimental approaches to linear regression to demonstrate different concepts.

### 1. Simple Linear Regression (Forecasting)
*   **Goal**: Predict today's mean temperature (`TG`) using *only* yesterday's mean temperature (`TG_lag1`).
*   **Type**: Time series forecasting.
*   **Validity**: This is a valid predictive approach as it uses past data to predict the future.

### 2. Multiple Linear Regression (Correlation Analysis)
*   **Goal**: Predict mean temperature (`TG`) using Minimum (`TN`) and Maximum (`TX`) temperatures, and the Mean temperature (`TG`) of the *same day*.
*   **Observation**: By including the target variable (`TG`) or its direct components within the predictors, the model achieves near-perfect performance.
*   **Note on Data Leakage**: In a real-world predictive scenario where we want to know *tomorrow's* temperature, this approach would represent **data leakage**, as we wouldn't have access to tomorrow's `TN`, `TX`, or `TG` yet. However, this serves as a mathematical demonstration of the correlation between these variables.

## Results

The following table summarizes the performance of the models on the test set (2023-2025):

| Experiment | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- |
| **Simple Regression** (TG_lag1 -> TG) | 17.613 | 22.615 | 0.880 |
| **Multiple Regression** (TN, TX, TG -> TG) | 0.000 | 0.000 | 1.000 |

### Visualizations

#### Simple Linear Regression: Real vs Predicted
![Simple Regression Results](figures/tg_simple_real_vs_predicho.png)

#### Multiple Linear Regression: Real vs Predicted
![Multiple Regression Results](figures/tg_multi_real_vs_predicho.png)

## Execution

Run `main.py` to load data, train models, and generate results and visualizations:

```bash
python main.py
```

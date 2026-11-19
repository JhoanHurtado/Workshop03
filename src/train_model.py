"""
train_model.py - Entrenamiento, validación y serialización del modelo de regresión.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    RAW_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
    FEATURE_COLS, TARGET_COL, SQLITE_DB_PATH
)
from src.schema_mapper import standardize_dataframe
from src.database import init_database, save_warehouse_records


def load_all_datasets() -> pd.DataFrame:
    """Carga y unifica los 5 conjuntos de datos (2015-2019)."""
    dfs = []
    for year in [2015, 2016, 2017, 2018, 2019]:
        file_path = RAW_DATA_DIR / f"{year}.csv"
        if file_path.exists():
            df_year = pd.read_csv(file_path)
            df_std = standardize_dataframe(df_year, year=year)
            dfs.append(df_std)
        else:
            print(f"⚠️ Archivo no encontrado: {file_path}")

    if not dfs:
        raise FileNotFoundError("No se encontraron archivos en data/raw/.")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def compute_metrics(y_true, y_pred, n_features: int) -> dict:
    """Calcula MAE, MSE, RMSE, R² y R² Ajustado."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Adjusted_R2": float(adj_r2)
    }


def train_and_evaluate():
    """Entrena los modelos, genera reportes visuales y serializa el modelo principal."""
    print("🔹 Cargando y unificando datasets 2015-2019...")
    df = load_all_datasets()
    print(f"   Total registros combinados: {len(df)}")

    # Guardar en base de datos SQLite
    init_database()
    save_warehouse_records(df, table_name="happiness_warehouse", if_exists="replace")
    print(f"✅ Registros guardados en tabla 'happiness_warehouse' de SQLite.")

    # Separar características y variable objetivo
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # División entrenamiento / prueba (80/20) con semilla fija
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"   Train set: {len(X_train)} filas | Test set: {len(X_test)} filas")

    # Definir pipelines de modelado (evitando data leakage con Imputer y Scaler dentro del Pipeline)
    pipe_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    pipe_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Entrenar Regresión Lineal
    pipe_lr.fit(X_train, y_train)
    y_pred_lr = pipe_lr.predict(X_test)
    metrics_lr = compute_metrics(y_test, y_pred_lr, len(FEATURE_COLS))

    # Entrenar Random Forest para comparación
    pipe_rf.fit(X_train, y_train)
    y_pred_rf = pipe_rf.predict(X_test)
    metrics_rf = compute_metrics(y_test, y_pred_rf, len(FEATURE_COLS))

    print("\n📊 RESULTADOS DE EVALUACIÓN (TEST SET):")
    print("--------------------------------------------------")
    print(f"{'Métrica':<15} | {'Linear Regression':<18} | {'Random Forest':<18}")
    print("--------------------------------------------------")
    for m in ["MAE", "MSE", "RMSE", "R2", "Adjusted_R2"]:
        print(f"{m:<15} | {metrics_lr[m]:<18.4f} | {metrics_rf[m]:<18.4f}")
    print("--------------------------------------------------")

    # Coeficientes del modelo de Regresión Lineal
    lr_coefs = pipe_lr.named_steps["regressor"].coef_
    intercept = pipe_lr.named_steps["regressor"].intercept_
    print(f"\nIntercepto (β0): {intercept:.4f}")
    print("Coeficientes estandarizados:")
    coef_dict = {}
    for col, coef in zip(FEATURE_COLS, lr_coefs):
        coef_dict[col] = float(coef)
        print(f"  • {col:<26}: {coef:+.4f}")

    # Guardar modelo principal serializado
    model_path = MODELS_DIR / "happiness_linear_model.joblib"
    joblib.dump(pipe_lr, model_path)
    print(f"\n💾 Modelo serializado guardado en: {model_path}")

    # Guardar métricas en JSON
    metrics_data = {
        "linear_regression": metrics_lr,
        "random_forest": metrics_rf,
        "coefficients": coef_dict,
        "intercept": float(intercept),
        "features": FEATURE_COLS
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)

    # 1. Gráfica de Real vs Predicho
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_lr, alpha=0.7, color="#2b5c8f", edgecolors="k", label="Observaciones Test")
    min_val, max_val = min(y_test.min(), y_pred_lr.min()), max(y_test.max(), y_pred_lr.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Línea Ideal (y = x)")
    plt.title(f"Regresión Lineal: Valores Reales vs. Predichos (R² = {metrics_lr['R2']:.3f})", fontsize=12)
    plt.xlabel("Puntaje Real de Felicidad (Actual)", fontsize=11)
    plt.ylabel("Puntaje Predicho (Predicted)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_predictions_vs_actual.png", dpi=300)
    plt.close()

    # 2. Gráfica de Residuos
    residuals = y_test - y_pred_lr
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred_lr, residuals, alpha=0.7, color="#d95f02", edgecolors="k")
    axes[0].axhline(0, color="black", linestyle="--", lw=1.5)
    axes[0].set_title("Residuos vs. Valores Predichos", fontsize=12)
    axes[0].set_xlabel("Predicción", fontsize=11)
    axes[0].set_ylabel("Residuo (y_test - y_pred)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    sns.histplot(residuals, kde=True, ax=axes[1], color="#7570b3")
    axes[1].set_title("Distribución de Residuos (Normalidad)", fontsize=12)
    axes[1].set_xlabel("Residuo", fontsize=11)
    axes[1].set_ylabel("Frecuencia", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_residuals_plot.png", dpi=300)
    plt.close()

    # 3. Importancia / Magnitud de Coeficientes
    plt.figure(figsize=(9, 5))
    s_coefs = pd.Series(lr_coefs, index=FEATURE_COLS).sort_values()
    s_coefs.plot(kind="barh", color="#1b9e77", edgecolor="black")
    plt.title("Impacto de Cada Variable en el Puntaje de Felicidad (Coeficientes Estandarizados)", fontsize=12)
    plt.xlabel("Coeficiente de Regresión Lineal", fontsize=11)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=300)
    plt.close()

    print(f"📊 Gráficas de evaluación generadas en: {FIGURES_DIR}")
    return pipe_lr, metrics_lr


if __name__ == "__main__":
    train_and_evaluate()

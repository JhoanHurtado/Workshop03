"""
eda.py - Análisis Exploratorio de Datos (EDA) y generación de visualizaciones.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, REPORTS_DIR, FEATURE_COLS, TARGET_COL
from src.database import load_warehouse_data


def run_eda():
    """Genera el análisis descriptivo y visualizaciones de los datos consolidados."""
    print("🔹 Ejecutando Análisis Exploratorio de Datos (EDA)...")
    df = load_warehouse_data()
    
    if df.empty:
        raise ValueError("La tabla 'happiness_warehouse' está vacía. Ejecute train_model.py primero.")

    # 1. Estadísticas descriptivas
    desc = df[[TARGET_COL] + FEATURE_COLS].describe()
    print("\n📈 Resumen estadístico:")
    print(desc)
    
    desc_dict = desc.to_dict()
    with open(REPORTS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(desc_dict, f, indent=4)

    # 2. Matriz de correlación
    plt.figure(figsize=(9, 7))
    corr = df[[TARGET_COL] + FEATURE_COLS].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Coeficiente Pearson'})
    plt.title("Matriz de Correlación de Variables del World Happiness Report", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_correlation_matrix.png", dpi=300)
    plt.close()

    # 3. Distribución de la variable objetivo (Happiness Score)
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET_COL], kde=True, color="#2c7fb8", bins=20)
    plt.axvline(df[TARGET_COL].mean(), color="red", linestyle="--", label=f"Media: {df[TARGET_COL].mean():.2f}")
    plt.axvline(df[TARGET_COL].median(), color="green", linestyle=":", label=f"Mediana: {df[TARGET_COL].median():.2f}")
    plt.title("Distribución del Puntaje de Felicidad (2015-2019)", fontsize=12)
    plt.xlabel("Happiness Score", fontsize=11)
    plt.ylabel("Frecuencia", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_target_distribution.png", dpi=300)
    plt.close()

    # 4. Relación de variables clave con el Puntaje de Felicidad
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    key_vars = [
        ("gdp_per_capita", "PIB per cápita (GDP)", "#1f78b4"),
        ("social_support", "Apoyo Social (Social Support)", "#33a02c"),
        ("healthy_life_expectancy", "Esperanza de Vida (Life Expectancy)", "#e31a1c")
    ]
    for idx, (col, label, color) in enumerate(key_vars):
        sns.regplot(data=df, x=col, y=TARGET_COL, ax=axes[idx], color=color, scatter_kws={'alpha': 0.6})
        axes[idx].set_title(f"{label} vs {TARGET_COL}", fontsize=11)
        axes[idx].set_xlabel(label, fontsize=10)
        axes[idx].set_ylabel("Happiness Score", fontsize=10)
        axes[idx].grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_key_relationships.png", dpi=300)
    plt.close()

    # 5. Evolución temporal del puntaje promedio por año
    plt.figure(figsize=(8, 5))
    yearly_avg = df.groupby("year")[TARGET_COL].mean().reset_index()
    sns.lineplot(data=yearly_avg, x="year", y=TARGET_COL, marker="o", color="#e7298a", linewidth=2.5)
    plt.title("Evolución del Puntaje Promedio de Felicidad (2015-2019)", fontsize=12)
    plt.xlabel("Año", fontsize=11)
    plt.ylabel("Puntaje Promedio", fontsize=11)
    plt.xticks([2015, 2016, 2017, 2018, 2019])
    plt.ylim(5.0, 5.8)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_yearly_trend.png", dpi=300)
    plt.close()

    print(f"✅ Gráficas EDA guardadas exitosamente en: {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()

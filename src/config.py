"""
config.py - Parámetros de configuración del proyecto ETL & Machine Learning.
"""

import os
from pathlib import Path

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DB_DIR = BASE_DIR / "db"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Crear directorios necesarios
for d in [DATA_DIR, RAW_DATA_DIR, DB_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Base de datos SQLite
SQLITE_DB_PATH = DB_DIR / "happiness.db"

# Parámetros de Kafka
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka-classroom-2026.westus.cloudapp.azure.com:9092")
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL")
KAFKA_SASL_MECHANISM = os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512")
KAFKA_SASL_USERNAME = os.getenv("KAFKA_SASL_USERNAME", "classroom-students")
KAFKA_SASL_PASSWORD = os.getenv("KAFKA_SASL_PASSWORD", "")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "etl.records")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "jhoan-hurtado")

# Definición canónica de características (features)
# Ordenado por fuerza de correlación con happiness_score
FEATURE_COLS = [
    "gdp_per_capita",
    "healthy_life_expectancy",
    "social_support",
    "freedom",
    "perceptions_of_corruption",
    "generosity"
]

TARGET_COL = "happiness_score"  # variable objetivo principal

CANONICAL_COLS = ["country", "year", TARGET_COL] + FEATURE_COLS

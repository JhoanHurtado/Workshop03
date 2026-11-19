"""
database.py - Gestor de almacenamiento en base de datos SQLite.
"""

import sqlite3
import pandas as pd
from typing import List, Dict, Any
from src.config import SQLITE_DB_PATH, CANONICAL_COLS


def get_connection():
    """Retorna una conexión a la base de datos SQLite."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    return conn


def init_database():
    """Crea las tablas necesarias en la base de datos si no existen."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Tabla para auditoría y registro de mensajes en bruto recibidos de Kafka
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_kafka_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                partition INTEGER,
                msg_offset INTEGER,
                timestamp INTEGER,
                payload TEXT,
                received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Tabla del almacén de datos (Warehouse) con datos estructurados
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS happiness_warehouse (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country TEXT,
                year INTEGER,
                happiness_score REAL,
                gdp_per_capita REAL,
                social_support REAL,
                healthy_life_expectancy REAL,
                freedom REAL,
                generosity REAL,
                perceptions_of_corruption REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Tabla para resultados de inferencia en tiempo real (Streaming Predictions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS streaming_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country TEXT,
                year INTEGER,
                actual_happiness_score REAL,
                predicted_happiness_score REAL,
                absolute_error REAL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.commit()
    print(f"✅ Base de datos inicializada correctamente en: {SQLITE_DB_PATH}")


def save_raw_message(topic: str, partition: int, offset: int, timestamp: int, payload: str):
    """Guarda un mensaje sin procesar en la tabla raw_kafka_messages."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO raw_kafka_messages (topic, partition, msg_offset, timestamp, payload)
            VALUES (?, ?, ?, ?, ?)
        """, (topic, partition, offset, timestamp, payload))
        conn.commit()


def save_warehouse_records(df: pd.DataFrame, table_name: str = "happiness_warehouse", if_exists: str = "append"):
    """Guarda un DataFrame procesado en la tabla correspondiente de SQLite."""
    with get_connection() as conn:
        df[CANONICAL_COLS].to_sql(table_name, conn, if_exists=if_exists, index=False)


def save_streaming_prediction(country: str, year: int, actual: float, pred: float):
    """Registra una predicción generada en streaming con su error."""
    abs_err = abs(actual - pred) if actual is not None and pred is not None else None
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO streaming_predictions (country, year, actual_happiness_score, predicted_happiness_score, absolute_error)
            VALUES (?, ?, ?, ?, ?)
        """, (country, year, actual, pred, abs_err))
        conn.commit()


def load_warehouse_data() -> pd.DataFrame:
    """Carga los datos consolidados desde la base de datos."""
    with get_connection() as conn:
        df = pd.read_sql("SELECT * FROM happiness_warehouse", conn)
    return df

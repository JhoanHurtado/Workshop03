#!/usr/bin/env python3
"""
run_pipeline.py - Pipeline principal de ejecución del sistema ETL & Machine Learning.

Ejecuta el flujo completo de forma reproducible:
1. Carga y estandarización de datos (2015-2019).
2. Almacenamiento en base de datos SQLite (Data Warehouse).
3. Análisis Exploratorio de Datos (EDA) y gráficos.
4. Entrenamiento del modelo Scikit-learn (evitando data leakage) y evaluación.
5. Ingesta / Consumo en streaming con inferencia y persistencia.
"""

import sys
import argparse
from src.database import init_database
from src.eda import run_eda
from src.train_model import train_and_evaluate
from src.kafka_consumer import HappinessStreamingConsumer


def main():
    parser = argparse.ArgumentParser(description="Pipeline ETL + Machine Learning: World Happiness")
    parser.add_argument("--password", type=str, default=None, help="Contraseña SASL para Kafka (opcional)")
    parser.add_argument("--group-id", type=str, default=None, help="Consumer Group ID")
    parser.add_argument("--max-messages", type=int, default=30, help="Mensajes máximos para consumir")
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout de Kafka en segundos")
    parser.add_argument("--simulate", action="store_true", help="Forzar simulación de streaming")
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 INICIANDO PIPELINE ETL KAFKA + MACHINE LEARNING")
    print("=" * 70)

    # Paso 1: Base de datos
    print("\n[PASO 1/4] Inicializando Base de Datos SQLite...")
    init_database()

    # Paso 2: Entrenamiento y Almacén
    print("\n[PASO 2/4] Entrenando Modelos de Machine Learning & Cargando Warehouse...")
    model, metrics = train_and_evaluate()

    # Paso 3: Análisis Exploratorio de Datos
    print("\n[PASO 3/4] Generando Análisis Exploratorio de Datos (EDA)...")
    run_eda()

    # Paso 4: Consumo en Streaming e Inferencia
    print("\n[PASO 4/4] Ejecutando Consumo de Streaming con Inferencia en Vivo...")
    consumer = HappinessStreamingConsumer(password=args.password, group_id=args.group_id)
    if args.simulate or not consumer.password:
        consumer.simulate_stream(max_messages=args.max_messages)
    else:
        consumer.consume_stream(max_messages=args.max_messages, timeout_sec=args.timeout)

    print("\n" + "=" * 70)
    print("✨ PIPELINE COMPLETADO EXITOSAMENTE")
    print("  • Base de datos: db/happiness.db")
    print("  • Modelo entrenado: models/happiness_linear_model.joblib")
    print("  • Gráficas y métricas: reports/figures/ y reports/metrics.json")
    print("=" * 70)


if __name__ == "__main__":
    main()

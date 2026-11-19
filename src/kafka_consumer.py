"""
kafka_consumer.py - Consumidor de Kafka para extracción, inferencia y carga en base de datos.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd
import joblib
import certifi

from confluent_kafka import Consumer, KafkaError, KafkaException

from src.config import (
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_SECURITY_PROTOCOL, KAFKA_SASL_MECHANISM,
    KAFKA_SASL_USERNAME, KAFKA_SASL_PASSWORD, KAFKA_TOPIC, KAFKA_GROUP_ID,
    MODELS_DIR, FEATURE_COLS, TARGET_COL, RAW_DATA_DIR
)
from src.schema_mapper import normalize_record
from src.database import (
    init_database, save_raw_message, save_streaming_prediction, save_warehouse_records
)


class HappinessStreamingConsumer:
    """
    Consumidor de Kafka que acumula registros en memoria, normaliza el esquema,
    ejecuta inferencia de Machine Learning y persiste en SQLite.
    """

    def __init__(self, password: Optional[str] = None, group_id: Optional[str] = None):
        self.password = password or os.getenv("KAFKA_SASL_PASSWORD") or KAFKA_SASL_PASSWORD
        self.group_id = group_id or KAFKA_GROUP_ID
        self.in_memory_records: List[Dict[str, Any]] = []
        
        # Cargar modelo de inferencia si existe
        self.model = None
        model_path = MODELS_DIR / "happiness_linear_model.joblib"
        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                print(f"🤖 Modelo de inferencia cargado desde: {model_path}")
            except Exception as e:
                print(f"⚠️ No se pudo cargar el modelo de ML: {e}")
        else:
            print("⚠️ Modelo no encontrado en models/. Ejecutando solo extracción.")

        init_database()

    def get_kafka_consumer(self) -> Optional[Consumer]:
        """Inicializa el cliente de confluent-kafka."""
        if not self.password:
            print("ℹ️ No se ha configurado 'KAFKA_SASL_PASSWORD'.")
            return None

        conf = {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "security.protocol": KAFKA_SECURITY_PROTOCOL,
            "sasl.mechanism": KAFKA_SASL_MECHANISM,
            "sasl.username": KAFKA_SASL_USERNAME,
            "sasl.password": self.password,
            "ssl.ca.location": certifi.where(),
            "group.id": self.group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
            "session.timeout.ms": 10000,
        }

        try:
            consumer = Consumer(conf)
            consumer.subscribe([KAFKA_TOPIC])
            print(f"✅ Conectado a Kafka broker: {KAFKA_BOOTSTRAP_SERVERS}")
            print(f"   Topic suscrito: '{KAFKA_TOPIC}' | Group ID: '{self.group_id}'")
            return consumer
        except Exception as e:
            print(f"❌ Error al conectar a Kafka: {e}")
            return None

    def process_message(self, topic: str, partition: int, offset: int, timestamp: int, payload_str: str):
        """
        Procesa individualmente un mensaje:
        1. Guarda mensaje crudo en base de datos.
        2. Normaliza el registro al esquema canónico.
        3. Realiza predicción con el modelo de ML.
        4. Acumula en el estado en memoria.
        5. Guarda la predicción en la base de datos.
        """
        # 1. Auditoría raw
        save_raw_message(topic, partition, offset, timestamp, payload_str)

        # 2. Deserialización JSON
        try:
            raw_data = json.loads(payload_str)
        except json.JSONDecodeError:
            print(f"⚠️ Mensaje no es JSON válido (Offset {offset}): {payload_str[:50]}")
            return

        # 3. Normalización
        norm_rec = normalize_record(raw_data)
        self.in_memory_records.append(norm_rec)

        # 4. Inferencia con Scikit-learn
        country = norm_rec.get("country", "Unknown")
        year = norm_rec.get("year", None)
        actual_score = norm_rec.get(TARGET_COL, None)
        predicted_score = None

        if self.model is not None:
            # Crear DataFrame de 1 fila para inferencia
            feature_row = pd.DataFrame([{col: norm_rec.get(col) for col in FEATURE_COLS}])
            try:
                pred = self.model.predict(feature_row)
                predicted_score = float(pred[0])
            except Exception as e:
                # Manejo de error si faltan columnas
                predicted_score = None

        # 5. Guardar predicción en SQLite
        if predicted_score is not None:
            save_streaming_prediction(country, year, actual_score, predicted_score)

        return norm_rec, predicted_score

    def consume_stream(self, max_messages: int = 100, timeout_sec: float = 10.0):
        """Consume mensajes reales del topic de Kafka."""
        consumer = self.get_kafka_consumer()
        if consumer is None:
            print("⚠️ Operando en modo simulación debido a credenciales no disponibles.")
            return self.simulate_stream(max_messages=max_messages)

        print(f"\n📡 Iniciando consumo en streaming (máx: {max_messages} mensajes, timeout: {timeout_sec}s)...")
        consumed_count = 0
        start_time = time.time()

        try:
            while consumed_count < max_messages:
                # Chequear timeout general
                if (time.time() - start_time) > timeout_sec:
                    print(f"⏱️ Tiempo de espera ({timeout_sec}s) alcanzado.")
                    break

                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        print(f"🏁 Fin de partición alcanzado: {msg.partition()}")
                        break
                    else:
                        print(f"❌ Error en mensaje Kafka: {msg.error()}")
                        break

                topic = msg.topic()
                partition = msg.partition()
                offset = msg.offset()
                ts_type, ts = msg.timestamp()
                payload = msg.value().decode("utf-8")

                norm_rec, pred = self.process_message(topic, partition, offset, ts, payload)
                consumed_count += 1

                actual_str = f"{norm_rec.get(TARGET_COL):.3f}" if norm_rec.get(TARGET_COL) is not None else "N/A"
                pred_str = f"{pred:.3f}" if pred is not None else "N/A"
                print(f"[{consumed_count}] Offset {offset} | País: {norm_rec.get('country')} | Actual: {actual_str} | Pred: {pred_str}")

        except KeyboardInterrupt:
            print("\n🛑 Consumo detenido manualmente por el usuario.")
        finally:
            consumer.close()
            print(f"🔌 Consumidor cerrado. Total mensajes consumidos: {consumed_count}")

        return self.get_in_memory_dataframe()

    def simulate_stream(self, max_messages: int = 50):
        """
        Simulación de streaming: Transmite filas de los datasets 2015-2019
        emulando la recepción de eventos de Kafka.
        """
        print(f"\n🧪 Iniciando simulación de streaming ({max_messages} eventos)...")
        csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
        if not csv_files:
            print("❌ No hay archivos CSV en data/raw/ para simular.")
            return pd.DataFrame()

        count = 0
        for f in csv_files:
            df = pd.read_csv(f)
            # Agregar columna de año si no existe
            year = int(f.stem) if f.stem.isdigit() else 2015
            df["year"] = year
            for _, row in df.iterrows():
                if count >= max_messages:
                    break
                payload = row.to_json()
                norm_rec, pred = self.process_message(
                    topic=KAFKA_TOPIC,
                    partition=0,
                    offset=count,
                    timestamp=int(time.time() * 1000),
                    payload_str=payload
                )
                count += 1
                actual_str = f"{norm_rec.get(TARGET_COL):.3f}" if norm_rec.get(TARGET_COL) is not None else "N/A"
                pred_str = f"{pred:.3f}" if pred is not None else "N/A"
                print(f"[Sim-{count}] Offset {count-1} | País: {norm_rec.get('country')} ({year}) | Actual: {actual_str} | Pred: {pred_str}")
                time.sleep(0.01)

            if count >= max_messages:
                break

        print(f"✅ Simulación finalizada. Total eventos procesados: {count}")
        return self.get_in_memory_dataframe()

    def get_in_memory_dataframe(self) -> pd.DataFrame:
        """Retorna el estado acumulado en memoria como un DataFrame."""
        if not self.in_memory_records:
            return pd.DataFrame()
        return pd.DataFrame(self.in_memory_records)


def main():
    parser = argparse.ArgumentParser(description="Consumidor Kafka con Inferencia de Machine Learning")
    parser.add_argument("--password", type=str, default=None, help="Contraseña SASL para Kafka")
    parser.add_argument("--group-id", type=str, default=None, help="Consumer Group ID")
    parser.add_argument("--max-messages", type=int, default=50, help="Máximo de mensajes a consumir")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout de espera en segundos")
    parser.add_argument("--simulate", action="store_true", help="Forzar modo simulación local")
    args = parser.parse_args()

    consumer = HappinessStreamingConsumer(password=args.password, group_id=args.group_id)
    if args.simulate:
        df_state = consumer.simulate_stream(max_messages=args.max_messages)
    else:
        df_state = consumer.consume_stream(max_messages=args.max_messages, timeout_sec=args.timeout)

    print("\n📦 Estado acumulado en memoria:")
    print(f"Dimensiones: {df_state.shape}")
    if not df_state.empty:
        print(df_state.head())


if __name__ == "__main__":
    main()

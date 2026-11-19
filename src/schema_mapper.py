"""
schema_mapper.py - Normalización y mapeo de esquemas heterogéneos (2015-2019).
"""

from typing import Dict, Any, Optional
import pandas as pd
from src.config import FEATURE_COLS, TARGET_COL, CANONICAL_COLS

COLUMN_MAPPING_RULES = {
    # País / Región
    "country": "country",
    "country or region": "country",
    
    # Puntaje de felicidad
    "happiness score": "happiness_score",
    "happiness.score": "happiness_score",
    "score": "happiness_score",
    
    # Economía / PIB
    "economy (gdp per capita)": "gdp_per_capita",
    "economy..gdp.per.capita.": "gdp_per_capita",
    "gdp per capita": "gdp_per_capita",
    
    # Apoyo social / Familia
    "family": "social_support",
    "social support": "social_support",
    
    # Esperanza de vida / Salud
    "health (life expectancy)": "healthy_life_expectancy",
    "health..life.expectancy.": "healthy_life_expectancy",
    "healthy life expectancy": "healthy_life_expectancy",
    
    # Libertad
    "freedom": "freedom",
    "freedom to make life choices": "freedom",
    
    # Generosidad
    "generosity": "generosity",
    
    # Percepción de corrupción / Confianza
    "trust (government corruption)": "perceptions_of_corruption",
    "trust..government.corruption.": "perceptions_of_corruption",
    "perceptions of corruption": "perceptions_of_corruption",
}


def normalize_record(raw_dict: Dict[str, Any], default_year: Optional[int] = None) -> Dict[str, Any]:
    """
    Normaliza un registro con nombres de columnas heterogéneas al esquema canónico.
    """
    clean_rec = {}
    
    # Limpieza de nombres de claves (minúsculas, trim)
    lower_dict = {str(k).strip().lower(): v for k, v in raw_dict.items()}
    
    for raw_k, raw_v in lower_dict.items():
        if raw_k in COLUMN_MAPPING_RULES:
            canonical_k = COLUMN_MAPPING_RULES[raw_k]
            clean_rec[canonical_k] = raw_v
            
    # Asignar año si está presente o usar default_year
    if "year" in lower_dict:
        try:
            clean_rec["year"] = int(lower_dict["year"])
        except (ValueError, TypeError):
            clean_rec["year"] = default_year
    elif default_year:
        clean_rec["year"] = default_year
    else:
        clean_rec["year"] = None

    # Asegurar que todas las columnas numéricas sean float
    numeric_cols = [TARGET_COL] + FEATURE_COLS
    for col in numeric_cols:
        if col in clean_rec:
            try:
                clean_rec[col] = float(clean_rec[col])
            except (ValueError, TypeError):
                clean_rec[col] = None
        else:
            clean_rec[col] = None

    if "country" not in clean_rec and "country" in lower_dict:
        clean_rec["country"] = str(lower_dict["country"])
        
    return clean_rec


def standardize_dataframe(df: pd.DataFrame, year: Optional[int] = None) -> pd.DataFrame:
    """
    Convierte un DataFrame de cualquier año (2015-2019) a las columnas canónicas.
    """
    records = df.to_dict(orient="records")
    normalized = [normalize_record(r, default_year=year) for r in records]
    df_norm = pd.DataFrame(normalized)
    
    # Asegurar orden canónico
    for col in CANONICAL_COLS:
        if col not in df_norm.columns:
            df_norm[col] = None
            
    return df_norm[CANONICAL_COLS]

"""Data preparation and audit helpers for the Lab 7 notebook.

Excel is read one file at a time. Every analytical operation after conversion is
performed with Spark, including missingness, exclusions, and duplicate checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COLUMNS = [
    "ANIO", "TRIMESTRE", "DOMINIO", "NUM_HOGAR", "FACTOR",
    "NUM_PERSONA", "P02A03", "P05C07A", "P05C07B", "P05H01A",
    "P03A03A", "P05C16", "OCUPADOS", "P05D01",
]
NUMERIC_FIELDS = {
    "P02A03": "edad",
    "P05C07A": "antiguedad_anios",
    "P05C07B": "antiguedad_meses",
    "P05H01A": "horas_semanales",
    "P05D01": "salario_mensual",
}
SELECTED_SCHEMA = T.StructType([T.StructField(name, T.StringType(), True) for name in SOURCE_COLUMNS])
KEY = ["periodo_archivo", "NUM_HOGAR", "NUM_PERSONA"]


def code(value: object) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value


def dictionary_codes(period: str, variable: str) -> dict[str, str]:
    path = ROOT / "data" / "dictionaries" / f"{period}_diccionario.xlsx"
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook.active.values)
    finally:
        workbook.close()
    starts = [i for i, row in enumerate(rows) if code(row[0]) == variable]
    if len(starts) < 2:
        raise ValueError(f"No se encontraron categorías para {variable} en {path}")
    result = {}
    for row in rows[starts[-1]:]:
        if row[0] is not None and code(row[0]) != variable:
            break
        if row[1] is not None and row[2] is not None:
            result[code(row[1])] = str(row[2]).strip()
    if not result:
        raise ValueError(f"Categorías vacías: {variable}, {path}")
    return result


def load_excel(spark: SparkSession, period: str) -> tuple[DataFrame, dict]:
    manifest_path = ROOT / "sources_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(x for x in manifest["archivos"] if x["periodo"] == period and x["tipo"] == "personas")
    path = ROOT / record["archivo"]
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.active
        actual_rows, actual_cols = sheet.max_row - 1, sheet.max_column
        headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    finally:
        workbook.close()
    expected = manifest["esperado"][period]
    if (actual_rows, actual_cols) != (expected["filas"], expected["columnas"]):
        raise ValueError(f"Dimensiones inesperadas en {period}: {(actual_rows, actual_cols)} vs {expected}")
    absent = set(SOURCE_COLUMNS) - set(headers)
    if absent:
        raise ValueError(f"Columnas faltantes en {period}: {sorted(absent)}")
    frame = pd.read_excel(path, usecols=lambda name: name in SOURCE_COLUMNS, engine="openpyxl", dtype=object)
    frame = frame[SOURCE_COLUMNS]
    if len(frame) != actual_rows:
        raise ValueError(f"Filas leídas {len(frame)} vs {actual_rows} en {period}")
    for column in SOURCE_COLUMNS:
        frame[column] = frame[column].map(lambda value: None if pd.isna(value) else str(value).strip())
    spark_frame = spark.createDataFrame(frame, schema=SELECTED_SCHEMA)
    year, trimester = int(period[:4]), int(period[-1])
    spark_frame = (spark_frame
                   .withColumn("archivo_origen", F.lit(path.name))
                   .withColumn("periodo_archivo", F.lit(period))
                   .withColumn("anio_archivo", F.lit(year))
                   .withColumn("trimestre_calendario", F.lit(trimester)))
    return spark_frame, {"filas": actual_rows, "columnas": actual_cols, "archivo": path.name}


def normalize_code(column: str):
    return F.regexp_replace(F.trim(F.col(column)), r"\.0+$", "")


def finite(column: str):
    value = F.col(column)
    return value.isNotNull() & ~F.isnan(value) & (F.abs(value) < F.lit(float("inf")))


def audit_duplicates(frame: DataFrame) -> dict:
    groups = frame.groupBy(*KEY).count().filter(F.col("count") > 1)
    duplicated_keys = groups.count()
    if not duplicated_keys:
        return {"claves_duplicadas": 0, "filas_duplicadas": 0, "conflictos": 0, "ejemplos": []}
    duplicated = frame.join(groups.select(*KEY), KEY, "inner")
    # Comparar todas las columnas del conjunto seleccionado, sin eliminar filas.
    hashes = duplicated.withColumn(
        "huella", F.sha2(F.to_json(F.struct(*[F.col(c) for c in frame.columns])), 256)
    )
    patterns = hashes.groupBy(*KEY).agg(F.count("*").alias("filas"), F.countDistinct("huella").alias("versiones"))
    result = patterns.agg(F.sum("filas").alias("filas"), F.sum((F.col("versiones") > 1).cast("int")).alias("conflictos")).first()
    examples = [r.asDict() for r in patterns.orderBy(F.desc("versiones"), *KEY).limit(10).collect()]
    return {"claves_duplicadas": duplicated_keys, "filas_duplicadas": result.filas,
            "conflictos": result.conflictos, "ejemplos": examples}


def missingness(frame: DataFrame, period: str) -> list[dict]:
    n = frame.count()
    exprs = [F.sum((F.col(c).isNull() | (F.trim(F.col(c)) == "")).cast("long")).alias(c)
             for c in SOURCE_COLUMNS]
    row = frame.agg(*exprs).first().asDict()
    return [{"periodo": period, "variable": c, "faltantes": int(row[c]),
             "porcentaje": round(100 * row[c] / n, 3)} for c in SOURCE_COLUMNS]


def prepare(frame: DataFrame, period: str) -> tuple[DataFrame, list[dict], list[dict]]:
    # Spark resolves names without regard to case by default; preserve the
    # source response before creating the analytical `dominio` column.
    normalized = frame.withColumn("DOMINIO_original", F.col("DOMINIO"))
    for source, target in NUMERIC_FIELDS.items():
        normalized = normalized.withColumn(target, F.col(source).cast("double"))
    for source, target in [("OCUPADOS", "ocupado"), ("P05C16", "categoria_ocupacional"),
                           ("P03A03A", "nivel_educativo"), ("DOMINIO", "dominio")]:
        normalized = normalized.withColumn(target, normalize_code(source))
    normalized = normalized.withColumn("antiguedad", F.col("antiguedad_anios") + F.col("antiguedad_meses") / 12)

    invalid_numeric = []
    for source, target in NUMERIC_FIELDS.items():
        counts = normalized.agg(
            F.sum((F.col(source).isNotNull() & (F.trim(F.col(source)) != "") & ~finite(target)).cast("long")).alias("invalidos")
        ).first()
        invalid_numeric.append({"periodo": period, "variable": source,
                                "invalidos_no_vacios": int(counts.invalidos or 0)})

    rules = [
        ("edad finita >= 15", finite("edad") & (F.col("edad") >= 15)),
        ("OCUPADOS = 1", F.col("ocupado") == "1"),
        ("P05C16 asalariado 1-4", F.col("categoria_ocupacional").isin("1", "2", "3", "4")),
        ("salario finito > 0", finite("salario_mensual") & (F.col("salario_mensual") > 0)),
        ("antiguedad_anios finita >= 0", finite("antiguedad_anios") & (F.col("antiguedad_anios") >= 0)),
        ("antiguedad_meses entero 0-11", finite("antiguedad_meses") &
         (F.col("antiguedad_meses") == F.floor("antiguedad_meses")) &
         F.col("antiguedad_meses").between(0, 11)),
        ("antiguedad <= edad", F.col("antiguedad") <= F.col("edad")),
        ("horas finitas 0-168", finite("horas_semanales") &
         (F.col("horas_semanales") > 0) & (F.col("horas_semanales") <= 168)),
    ]
    audit = []
    working = normalized
    previous = frame.count()
    for label, condition in rules:
        working = working.filter(condition)
        current = working.count()
        audit.append({"periodo": period, "paso": label, "entradas": previous,
                      "excluidos": previous - current, "restantes": current})
        previous = current

    for source, target in [("P03A03A", "nivel_educativo"), ("DOMINIO", "dominio")]:
        allowed = sorted(dictionary_codes(period, source))
        working = working.withColumn(target, F.when(F.col(target).isin(*allowed), F.col(target)).otherwise("DESCONOCIDO"))
    return working, audit, invalid_numeric


def stats(frame: DataFrame, field: str) -> dict:
    row = frame.agg(
        F.count(field).alias("n"), F.avg(field).alias("media"),
        F.stddev_samp(field).alias("desviacion"), F.min(field).alias("minimo"),
        F.max(field).alias("maximo"),
        F.expr(f"percentile_approx({field}, array(0.25, 0.5, 0.75, 0.95), 10000)").alias("percentiles"),
    ).first()
    result = row.asDict()
    result["p25"], result["mediana"], result["p75"], result["p95"] = result.pop("percentiles")
    return {"variable": field, **result}


def group_salary(frame: DataFrame, group: str) -> pd.DataFrame:
    return (frame.groupBy(group)
            .agg(F.count("*").alias("n"), F.expr("percentile_approx(salario_mensual, 0.5, 10000)").alias("salario_mediano"))
            .orderBy(group).toPandas())

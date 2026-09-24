"""Focused synthetic checks for the labor population rules."""

from pyspark.sql import SparkSession, functions as F

from scripts.analysis import SOURCE_COLUMNS, SELECTED_SCHEMA, audit_duplicates, prepare


def main() -> None:
    spark = (SparkSession.builder.master("local[2]").appName("Lab7-preparation-check")
             .config("spark.sql.shuffle.partitions", "2").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    base = dict.fromkeys(SOURCE_COLUMNS, None)
    base.update(ANIO="2025", TRIMESTRE="2", DOMINIO="1.0", NUM_HOGAR="1",
                NUM_PERSONA="1", FACTOR="100", P02A03="25", P05C07A="2",
                P05C07B="1", P05H01A="40", P03A03A="0", P05C16="1.0",
                OCUPADOS="1.0", P05D01="3000")
    cases = [base.copy()]
    edits = [
        {"P02A03": "14"}, {"P02A03": "Infinity"}, {"P05C16": "9"},
        {"P05D01": "0"}, {"P05C07A": "-1"}, {"P05C07B": "1.5"},
        {"P05C07A": "26"}, {"P05H01A": "169"}, {"P05H01A": None},
    ]
    for index, edit in enumerate(edits, 2):
        row = base.copy()
        row.update(edit, NUM_HOGAR=str(index))
        cases.append(row)
    frame = spark.createDataFrame([tuple(row[x] for x in SOURCE_COLUMNS) for row in cases], SELECTED_SCHEMA)
    frame = frame.withColumn("periodo_archivo", F.lit("2025T1"))
    clean, audit, invalid = prepare(frame, "2025T1")
    assert clean.count() == 1, audit
    survivor = clean.first()
    assert survivor.nivel_educativo == "0" and survivor.categoria_ocupacional == "1"
    assert survivor.dominio == "1" and abs(survivor.antiguedad - 2.0833333333) < 1e-6
    assert sum(x["excluidos"] for x in audit) == len(cases) - 1
    assert audit_duplicates(frame)["claves_duplicadas"] == 0
    exact = frame.unionByName(frame.filter(F.col("NUM_HOGAR") == "1"))
    assert audit_duplicates(exact)["claves_duplicadas"] == 1
    assert audit_duplicates(exact)["conflictos"] == 0
    conflict_row = base.copy()
    conflict_row["P05D01"] = "9999"
    conflict = spark.createDataFrame([tuple(conflict_row[x] for x in SOURCE_COLUMNS)], SELECTED_SCHEMA)
    conflict = conflict.withColumn("periodo_archivo", F.lit("2025T1"))
    assert audit_duplicates(exact.unionByName(conflict))["conflictos"] == 1
    print("Casos límite, orden de filtros y duplicados: correctos")
    spark.stop()


if __name__ == "__main__":
    main()

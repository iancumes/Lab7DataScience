"""Recheck the complete prepared datasets independently of notebook state."""

from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from scripts.analysis import finite


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    spark = (SparkSession.builder.master("local[2]").appName("Lab7-output-verification")
             .config("spark.sql.shuffle.partitions", "4").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    for name, period_set in [("eneic_2025.parquet", {"2025T1", "2025T2", "2025T3", "2025T4"}),
                             ("eneic_2026.parquet", {"2026T1"})]:
        data = spark.read.parquet(str(root / "data" / "processed" / name))
        assert {row.periodo_archivo for row in data.select("periodo_archivo").distinct().collect()} == period_set
        condition = (
            finite("edad") & (F.col("edad") >= 15) & (F.col("ocupado") == "1") &
            F.col("categoria_ocupacional").isin("1", "2", "3", "4") &
            finite("salario_mensual") & (F.col("salario_mensual") > 0) &
            finite("antiguedad_anios") & (F.col("antiguedad_anios") >= 0) &
            finite("antiguedad_meses") & F.col("antiguedad_meses").between(0, 11) &
            (F.col("antiguedad_meses") == F.floor("antiguedad_meses")) &
            (F.col("antiguedad") <= F.col("edad")) & finite("horas_semanales") &
            (F.col("horas_semanales") > 0) & (F.col("horas_semanales") <= 168)
        )
        invalid = data.filter(~condition | condition.isNull()).count()
        assert invalid == 0, (name, invalid)
        assert data.filter(F.col("nivel_educativo") == "0").count() > 0
        assert data.filter(~F.col("nivel_educativo").isin("0", "1", "2", "3", "4", "5", "6", "7", "DESCONOCIDO")).count() == 0
        assert data.filter(~F.col("dominio").isin("1", "2", "3", "DESCONOCIDO")).count() == 0
        assert data.columns.count("DOMINIO_original") == 1
        print(name, data.count(), "registros; cero violaciones de filtros", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()

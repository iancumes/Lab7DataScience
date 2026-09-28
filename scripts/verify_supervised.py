"""Recompute final test metrics from the persisted Spark ML pipelines."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoderModel, StringIndexerModel, VectorAssembler
from pyspark.sql import DataFrame, SparkSession, functions as F


ROOT = Path(__file__).resolve().parents[1]
PREDICTORS = {
    "edad", "antiguedad", "horas_semanales",
    "nivel_educativo", "categoria_ocupacional", "dominio",
}


def metrics(predictions: DataFrame) -> dict[str, float]:
    errors = predictions.select(
        (F.col("salario_mensual") - F.col("prediction")).alias("residuo")
    )
    row = errors.agg(
        F.avg(F.abs("residuo")).alias("MAE"),
        F.sqrt(F.avg(F.pow("residuo", 2))).alias("RMSE"),
    ).first()
    r2 = RegressionEvaluator(
        labelCol="salario_mensual", predictionCol="prediction", metricName="r2"
    ).evaluate(predictions)
    return {"MAE": float(row.MAE), "RMSE": float(row.RMSE), "R2": float(r2)}


def assert_pipeline_inputs(model: PipelineModel) -> None:
    indexers = [stage for stage in model.stages if isinstance(stage, StringIndexerModel)]
    encoders = [stage for stage in model.stages if isinstance(stage, OneHotEncoderModel)]
    assemblers = [stage for stage in model.stages if isinstance(stage, VectorAssembler)]
    assert {stage.getInputCol() for stage in indexers} == {
        "nivel_educativo", "categoria_ocupacional", "dominio"
    }
    assert len(encoders) == 1 and len(assemblers) == 1
    assembled = set(assemblers[0].getInputCols())
    assert {"edad", "antiguedad", "horas_semanales"}.issubset(assembled)
    original_inputs = {stage.getInputCol() for stage in indexers} | {
        name for name in assembled if not name.endswith("_ohe")
    }
    assert original_inputs == PREDICTORS


def main() -> None:
    artifact = json.loads((ROOT / "artifacts" / "supervised_metrics.json").read_text(encoding="utf-8"))
    spark = (SparkSession.builder.master("local[2]").appName("Lab7-supervised-verification")
             .config("spark.sql.shuffle.partitions", "4").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    test = spark.read.parquet(str(ROOT / "data" / "processed" / "eneic_2026.parquet")).cache()
    test_n = test.count()
    assert test_n == artifact["particiones"]["prueba_2026T1"]
    assert {row.periodo_archivo for row in test.select("periodo_archivo").distinct().collect()} == {"2026T1"}

    expected_rows = {row["modelo"]: row for row in artifact["prueba_2026T1"]}
    for model_name, directory in [
        ("Regresión lineal", "linear_final_2025"),
        ("Random Forest", "random_forest_final_2025"),
    ]:
        model = PipelineModel.load(str(ROOT / "artifacts" / "models" / directory))
        assert_pipeline_inputs(model)
        predictions = model.transform(test).cache()
        assert predictions.count() == test_n
        observed = metrics(predictions)
        expected = expected_rows[model_name]
        for metric, value in observed.items():
            assert math.isclose(value, float(expected[metric]), rel_tol=1e-9, abs_tol=1e-6), (
                model_name, metric, value, expected[metric]
            )
        predictions.unpersist()
        print(model_name, observed, flush=True)
    test.unpersist()
    spark.stop()
    print("Modelos finales, seis predictores y métricas 2026: correctos")


if __name__ == "__main__":
    main()

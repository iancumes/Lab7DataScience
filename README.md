# Laboratorio 7 — ENEIC con Spark MLlib

Entrega completa de los incisos 1–8 del Laboratorio 7 de CC3066 Data Science. El proyecto cubre preparación y auditoría, exploración, KMeans, regresión lineal, Random Forest, selección temporal de hiperparámetros, evaluación final en 2026 y análisis de errores. Todas las cifras principales son **no ponderadas** y describen los registros elegibles, no a la población guatemalteca.

## Reproducción

Requiere Docker y Docker Compose. Desde esta carpeta:

```powershell
docker compose build
docker compose run --rm lab7 python scripts/download_data.py
docker compose run --rm lab7 python scripts/build_notebook.py
docker compose run --rm lab7 jupyter nbconvert --execute --to notebook --inplace Laboratorio7_EDA_Segmentacion.ipynb --ExecutePreprocessor.timeout=3600
docker compose run --rm lab7 python -m scripts.check_preparation
docker compose run --rm lab7 python -m scripts.verify_outputs
docker compose run --rm lab7 python -m scripts.verify_supervised
docker compose up -d
```

Abrir `http://localhost:8889/lab` para revisar el notebook. La ejecución desde un kernel nuevo comienza por la primera celda y genera los Parquet, modelos y métricas. Los Excel se descargan desde la [página ENEIC oficial del INE](https://www.ine.gob.gt/encuesta-nacional-de-empleo-e-ingresos/); si el sitio sirve una página HTML en lugar de un Excel, el descargador se detiene y pide poner el archivo oficial en la ruta mostrada. No se inventan datos.

## Diseño de evaluación

- 2025T1–T3: entrenamiento y comparación de configuraciones.
- 2025T4: validación temporal y selección por menor RMSE.
- 2025T1–T4: reajuste final de la configuración elegida.
- 2026T1: prueba final, utilizada una sola vez después de la selección.
- Referencia: predicción constante igual al salario promedio del conjunto de ajuste.

## Estructura

- `Laboratorio7_EDA_Segmentacion.ipynb`: procedimiento completo, resultados ejecutados, gráficas e interpretación.
- `scripts/download_data.py`: fuentes oficiales, descarga y manifiesto de procedencia.
- `scripts/build_notebook.py`: fuente reproducible del notebook.
- `scripts/check_preparation.py`: pruebas sintéticas de reglas, orden de filtros y duplicados.
- `scripts/verify_outputs.py`: auditoría independiente de los Parquet preparados.
- `scripts/verify_supervised.py`: recálculo independiente de métricas desde los modelos persistidos.
- `data/raw/`: bases Personas originales (ignoradas por Git).
- `data/dictionaries/`: diccionarios originales (ignorados por Git).
- `data/processed/`: conjuntos Parquet preparados (ignorados por Git).
- `artifacts/`: KMeans, pipelines supervisados y métricas (ignorados por Git).
- `sources_manifest.json`: procedencia, hashes y dimensiones verificadas.

El repositorio conserva la guía de laboratorio como `Laboratorio 7.md`.

# Laboratorio 7 — ENEIC: exploración y segmentación

Avance de los incisos 1–4 del Laboratorio 7 de CC3066 Data Science. El análisis usa únicamente las bases **Personas** de ENEIC 2025 para exploración y KMeans; 2026T1 se prepara por separado para la etapa posterior. Todas las cifras principales son **no ponderadas** y describen los registros elegibles, no a la población guatemalteca.

## Reproducción

Requiere Docker y Docker Compose. Desde esta carpeta:

```powershell
docker compose build
docker compose run --rm lab7 python scripts/download_data.py
docker compose run --rm lab7 jupyter nbconvert --execute --to notebook --inplace Laboratorio7_EDA_Segmentacion.ipynb --ExecutePreprocessor.timeout=3600
docker compose run --rm lab7 python -m scripts.check_preparation
docker compose run --rm lab7 python -m scripts.verify_outputs
docker compose up -d
```

Abrir `http://localhost:8889/lab` para revisar el notebook. La ejecución desde un kernel nuevo comienza por la primera celda y genera los Parquet y artefactos de segmentación. Los Excel se descargan desde la [página ENEIC oficial del INE](https://www.ine.gob.gt/encuesta-nacional-de-empleo-e-ingresos/); si el sitio sirve una página HTML en lugar de un Excel, el descargador se detiene y pide poner el archivo oficial en la ruta mostrada. No se inventan datos.

## Estructura

- `Laboratorio7_EDA_Segmentacion.ipynb`: procedimiento, resultados, gráficas e interpretación.
- `scripts/download_data.py`: fuentes oficiales, descarga y manifiesto de procedencia.
- `data/raw/`: bases Personas originales (ignoradas por Git).
- `data/dictionaries/`: diccionarios originales (ignorados por Git).
- `data/processed/`: conjuntos Parquet preparados (ignorados por Git).
- `artifacts/`: modelo y asignaciones KMeans (ignorados por Git).
- `sources_manifest.json`: procedencia, hashes y dimensiones verificadas.

El repositorio conserva la guía de laboratorio como `Laboratorio 7.md`. Los archivos locales `PLAN_LAB7.md` y `PROGRESO_LAB7.md` están ignorados; para retomar en la misma máquina, leerlos y después consultar `git status --short`.

"""Rebuild the self-contained Lab 7 report notebook."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))


md("""
# Laboratorio 7 — ENEIC: Spark MLlib

**CC3066 Data Science · Entrega final · 27 de septiembre de 2026**

Este cuaderno estudia observaciones de personas asalariadas de 15 años o más con salario mensual positivo registrado. Los cuatro archivos de 2025 se usan para exploración, segmentación, selección y entrenamiento; 2026T1 se mantiene aislado para la evaluación final. Las cifras son **no ponderadas** y no representan estimaciones oficiales de Guatemala. El mismo individuo puede aportar observaciones en períodos distintos.

Fuente: [bases y diccionarios Personas ENEIC del INE](https://www.ine.gob.gt/encuesta-nacional-de-empleo-e-ingresos/). Procedencia y SHA-256 en `sources_manifest.json`. Ejecutar desde la primera celda en el contenedor del repositorio.
""")

md("""
## 0. Configuración y método

La lectura inicial de Excel se realiza archivo por archivo con pandas/openpyxl. Cada archivo se convierte enseguida en un DataFrame Spark de tipos explícitos; todos los filtros, estadísticas, correlaciones y modelos se calculan con Spark. Los gráficos reciben únicamente agregados o una muestra de hasta 5,000 registros. Los percentiles usan `percentile_approx` con precisión 10,000; la desviación estándar es muestral. Se conservan los salarios extremos y el objetivo en quetzales sin transformación.
""")
code(r'''
import json
import platform
from pathlib import Path
from IPython.display import display, Markdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark
%matplotlib inline
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.stat import Correlation
from scripts.analysis import (ROOT, SOURCE_COLUMNS, KEY, load_excel, dictionary_codes,
                              audit_duplicates, missingness, prepare, stats, group_salary)

spark = (SparkSession.builder.master("local[2]").appName("Lab7-ENEIC-Spark-MLlib")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.driver.memory", "3g")
         .config("spark.ui.showConsoleProgress", "false")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({"figure.figsize": (9, 4.8), "axes.titlesize": 13, "axes.labelsize": 11})
print({"Python": platform.python_version(), "PySpark": pyspark.__version__,
       "pandas": pd.__version__, "Matplotlib": plt.matplotlib.__version__,
       "Seaborn": sns.__version__, "Spark": spark.version})
''')

md("""
## 1. Carga, armonización y calidad de datos

Las dimensiones del archivo 2025T4 difieren de los otros trimestres (302 frente a 270 columnas). Por ello se seleccionan los mismos campos por nombre y se unen con `unionByName`; apilar por posición cambiaría el significado de columnas. `TRIMESTRE` se conserva como variable original. El trimestre calendario se asigna según el archivo publicado.
""")
code(r'''
manifest_path = ROOT / "sources_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
periodos = ["2025T1", "2025T2", "2025T3", "2025T4", "2026T1"]
display(pd.DataFrame([{
    "período": r["periodo"], "tipo": r["tipo"], "archivo": r["archivo"],
    "filas": r["filas_observadas"], "columnas": r["columnas_observadas"],
    "SHA-256": r["sha256"][:16] + "…"
} for r in manifest["archivos"]]))
assert all((next(r for r in manifest["archivos"] if r["periodo"] == p and r["tipo"] == "personas")["filas_observadas"],
            next(r for r in manifest["archivos"] if r["periodo"] == p and r["tipo"] == "personas")["columnas_observadas"])
           == (manifest["esperado"][p]["filas"], manifest["esperado"][p]["columnas"]) for p in periodos)
print("Diccionario 2026T1, encabezado interno:", next(r for r in manifest["archivos"] if r["periodo"] == "2026T1" and r["tipo"] == "diccionario")["encabezado_inicial"])
''')
code(r'''
diccionarios = {}
for periodo in periodos:
    diccionarios[periodo] = {nombre: dictionary_codes(periodo, nombre)
                            for nombre in ["DOMINIO", "P03A03A", "P05C16"]}
for nombre in ["DOMINIO", "P03A03A", "P05C16"]:
    print(nombre, diccionarios["2025T1"][nombre])
assert all("0" in diccionarios[p]["P03A03A"] for p in periodos)
assert all(set(["1", "2", "3", "4"]).issubset(diccionarios[p]["P05C16"]) for p in periodos)
''')
md("""
El valor educativo `0` figura como **NINGUNO** en los diccionarios; no se trata como faltante. Las categorías ausentes o ajenas al diccionario se convierten a `DESCONOCIDO` después del filtro numérico. Los campos de salario y antigüedad no se imputan.
""")
code(r'''
raws, prepared = {}, {}
loading, missing, conversions, filters, duplicate_audits, trimester_counts = [], [], [], [], [], []
for periodo in periodos:
    raw, info = load_excel(spark, periodo)
    raws[periodo] = raw.cache()
    loading.append({"periodo": periodo, **info, "registros_iniciales": raw.count()})
    print("Cargado", periodo, info)
    missing.extend(missingness(raw, periodo))
    duplicate_audits.append({"periodo": periodo, "etapa": "antes", **audit_duplicates(raw)})
    trimester_counts.extend([{ "periodo": periodo, "TRIMESTRE_original": r["TRIMESTRE"], "n": r["count"] }
                             for r in raw.groupBy("TRIMESTRE").count().collect()])
    clean, steps, invalids = prepare(raw, periodo)
    prepared[periodo] = clean.cache()
    clean.count()
    filters.extend(steps)
    conversions.extend(invalids)
    duplicate_audits.append({"periodo": periodo, "etapa": "después", **audit_duplicates(clean)})

print("Esquema de columnas seleccionadas y analíticas:")
prepared["2025T1"].printSchema()
display(prepared["2025T1"].select(*(SOURCE_COLUMNS + ["archivo_origen", "periodo_archivo", "anio_archivo", "trimestre_calendario", "edad", "antiguedad", "horas_semanales", "salario_mensual"])).limit(5).toPandas())
''')
code(r'''
loading_df = pd.DataFrame(loading)
missing_df = pd.DataFrame(missing)
conversions_df = pd.DataFrame(conversions)
filters_df = pd.DataFrame(filters)
duplicate_df = pd.DataFrame(duplicate_audits)
trimesters_df = pd.DataFrame(trimester_counts).sort_values(["periodo", "TRIMESTRE_original"])
display(loading_df[["periodo", "archivo", "filas", "columnas", "registros_iniciales"]])
display(trimesters_df)
display(missing_df.pivot(index="variable", columns="periodo", values=["faltantes", "porcentaje"]))
display(conversions_df)
display(filters_df)
display(duplicate_df[["periodo", "etapa", "claves_duplicadas", "filas_duplicadas", "conflictos"]])
for item in duplicate_audits:
    if item["claves_duplicadas"]:
        print(item["periodo"], item["etapa"], "ejemplos de claves:", item["ejemplos"])
assert not any(item["conflictos"] for item in duplicate_audits), "Hay claves duplicadas con registros en conflicto; investigar antes de continuar"
assert all(int(loading_df.loc[loading_df.periodo == p, "registros_iniciales"].iloc[0]) ==
           int(filters_df.loc[(filters_df.periodo == p) & (filters_df.paso == "edad finita >= 15"), "entradas"].iloc[0]) for p in periodos)
assert all(int(filters_df.loc[filters_df.periodo == p, "excluidos"].sum()) +
           int(filters_df.loc[filters_df.periodo == p, "restantes"].iloc[-1]) ==
           int(loading_df.loc[loading_df.periodo == p, "registros_iniciales"].iloc[0]) for p in periodos)
''')
code(r'''
train = prepared["2025T1"]
for periodo in periodos[1:4]:
    train = train.unionByName(prepared[periodo])
train = train.cache()
test_2026 = prepared["2026T1"].cache()
train_n, test_n = train.count(), test_2026.count()
assert train_n == sum(int(filters_df.loc[filters_df.periodo == p, "restantes"].iloc[-1]) for p in periodos[:4])
assert test_n == int(filters_df.loc[filters_df.periodo == "2026T1", "restantes"].iloc[-1])
processed_dir = ROOT / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)
train.write.mode("overwrite").parquet(str(processed_dir / "eneic_2025.parquet"))
test_2026.write.mode("overwrite").parquet(str(processed_dir / "eneic_2026.parquet"))
for name, expected_n in [("eneic_2025.parquet", train_n), ("eneic_2026.parquet", test_n)]:
    reread = spark.read.parquet(str(processed_dir / name))
    assert reread.count() == expected_n
    assert [(f.name, f.dataType.simpleString()) for f in reread.schema] == [
        (f.name, f.dataType.simpleString()) for f in train.schema]
print("Registros elegibles: 2025 =", train_n, "; 2026T1 =", test_n)
display(Markdown(f"De {sum(x['filas'] for x in loading[:4]):,} observaciones originales de 2025, quedaron **{train_n:,}** elegibles ({train_n / sum(x['filas'] for x in loading[:4]):.1%}). 2026T1 quedó separado con **{test_n:,}** registros elegibles. Los conteos son observaciones, no personas únicas ni trabajadores de todo el país."))
''')
md("""
### Interpretación de calidad y alcance

- Un campo vacío puede indicar que la pregunta no correspondía según el flujo de la encuesta, o que faltó registrar una respuesta. El código ausente por sí solo no permite distinguir esas causas; hay que revisar la boleta y el universo de la pregunta.
- La clave incorpora el período. Una persona observada en dos trimestres aporta dos observaciones longitudinales válidas; eliminarlas borraría información temporal.
- El subconjunto exige empleo asalariado y salario positivo registrado. Además, la ENEIC proviene de una muestra. El número de filas filtradas no es el total de trabajadores de Guatemala.
- `FACTOR` se conserva para un futuro análisis poblacional con el diseño muestral. Las tablas y modelos de este avance son deliberadamente no ponderados.
""")

md("""
## 2. Estadística descriptiva y exploración

Todos los resultados de esta sección se calculan sobre las observaciones elegibles de **2025**. Para los gráficos se transfieren a pandas únicamente tablas agregadas de Spark.
""")
code(r'''
numeric_cols = ["salario_mensual", "edad", "antiguedad", "horas_semanales"]
summary = pd.DataFrame([stats(train, name) for name in numeric_cols])
display(summary.round(2))
salary_row = summary.set_index("variable").loc["salario_mensual"]
display(Markdown(f"El salario mensual promedio es **Q{salary_row['media']:,.2f}** y la mediana **Q{salary_row['mediana']:,.2f}**; la diferencia es **Q{salary_row['media']-salary_row['mediana']:,.2f}**. El percentil 95 es **Q{salary_row['p95']:,.2f}** y el máximo **Q{salary_row['maximo']:,.2f}**. Una media superior a la mediana sugiere cola hacia salarios altos, que se contrasta con el histograma."))
''')
code(r'''
group_tables = {}
for column, label in [("categoria_ocupacional", "Categoría ocupacional"),
                      ("nivel_educativo", "Nivel educativo"), ("dominio", "Dominio")]:
    table = group_salary(train, column)
    table["porcentaje"] = 100 * table["n"] / train_n
    group_tables[column] = table.sort_values("n", ascending=False)
    display(Markdown(f"### {label}"))
    display(group_tables[column].round(2))
    fig, ax = plt.subplots()
    sns.barplot(data=group_tables[column], x=column, y="n", ax=ax, color="#2374ab")
    ax.set(title=f"Registros por {label.lower()} · asalariados elegibles, 2025", xlabel="Código", ylabel="Observaciones")
    plt.tight_layout(); plt.show()
    top = group_tables[column].iloc[0]
    display(Markdown(f"El grupo más frecuente es el código **{top[column]}**, con **{int(top['n']):,} registros ({top['porcentaje']:.1f}%)**. Las etiquetas de cada código se muestran arriba desde los diccionarios oficiales."))
''')
code(r'''
# Histograma agregado en Spark. Los intervalos son de 0.25 unidades log10.
hist = (train.withColumn("log_bin", F.floor(F.log10("salario_mensual") * 4) / 4)
        .groupBy("log_bin").count().orderBy("log_bin").toPandas())
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
axes[0].bar(hist["log_bin"], hist["count"], width=0.22, color="#2374ab")
axes[0].set(title="Distribución salarial · 2025", xlabel="log10(salario mensual en Q)", ylabel="Observaciones")
axes[1].bar(["Media", "Mediana"], [salary_row["media"], salary_row["mediana"]], color=["#2374ab", "#ef8354"])
axes[1].set(title="Media y mediana del salario · 2025", ylabel="Quetzales mensuales")
plt.tight_layout(); plt.show()
display(Markdown(f"La distribución abarca desde **Q{salary_row['minimo']:,.2f}** hasta **Q{salary_row['maximo']:,.2f}**. El eje del histograma está en **log10** solo para visualizar; las estadísticas y el salario modelado permanecen en quetzales originales. Los valores altos no fueron eliminados."))
''')
code(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, column, title in [(axes[0], "nivel_educativo", "Salario mediano por educación"),
                          (axes[1], "categoria_ocupacional", "Salario mediano por ocupación")]:
    table = group_tables[column].sort_values(column)
    sns.barplot(data=table, x=column, y="salario_mediano", ax=ax, color="#2a9d8f")
    ax.set(title=title + " · 2025", xlabel="Código", ylabel="Quetzales mensuales")
plt.tight_layout(); plt.show()
for column, name in [("nivel_educativo", "educación"), ("categoria_ocupacional", "ocupación")]:
    table = group_tables[column]
    highest = table.loc[table["salario_mediano"].idxmax()]
    lowest = table.loc[table["salario_mediano"].idxmin()]
    display(Markdown(f"Por {name}, la mediana mayor corresponde al código **{highest[column]}** (Q{highest['salario_mediano']:,.2f}; n={int(highest['n']):,}) y la menor al código **{lowest[column]}** (Q{lowest['salario_mediano']:,.2f}; n={int(lowest['n']):,}). Estas son asociaciones descriptivas, no efectos causales."))
''')
code(r'''
quarters = group_salary(train, "periodo_archivo").sort_values("periodo_archivo")
display(quarters)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
sns.lineplot(data=quarters, x="periodo_archivo", y="n", marker="o", ax=axes[0], color="#2374ab")
sns.lineplot(data=quarters, x="periodo_archivo", y="salario_mediano", marker="o", ax=axes[1], color="#d95f02")
axes[0].set(title="Tamaño analítico por trimestre · 2025", xlabel="Trimestre de archivo", ylabel="Observaciones")
axes[1].set(title="Salario mediano por trimestre · 2025", xlabel="Trimestre de archivo", ylabel="Quetzales mensuales")
plt.tight_layout(); plt.show()
first, last = quarters.iloc[0], quarters.iloc[-1]
display(Markdown(f"De {first['periodo_archivo']} a {last['periodo_archivo']}, el tamaño cambió de **{int(first['n']):,}** a **{int(last['n']):,}** observaciones y el salario mediano de **Q{first['salario_mediano']:,.2f}** a **Q{last['salario_mediano']:,.2f}**. Son cortes de archivo y no un panel balanceado de personas únicas."))
''')

md("""
## 3. Relaciones numéricas

Pearson mide asociación **lineal**. Se calcula sobre todos los registros elegibles de 2025 usando `VectorAssembler` y `Correlation.corr()` de Spark. Los extremos salariales pueden influir en los coeficientes; un coeficiente no demuestra causalidad.
""")
code(r'''
corr_cols = ["salario_mensual", "edad", "antiguedad", "horas_semanales"]
corr_input = VectorAssembler(inputCols=corr_cols, outputCol="corr_features").transform(train.select(*corr_cols))
matrix = Correlation.corr(corr_input, "corr_features", "pearson").first()[0].toArray()
corr_table = pd.DataFrame(matrix, index=corr_cols, columns=corr_cols)
display(corr_table.round(3))
fig, ax = plt.subplots(figsize=(6.6, 5.3))
sns.heatmap(corr_table, annot=True, fmt=".2f", vmin=-1, vmax=1, center=0, cmap="coolwarm", square=True, ax=ax)
ax.set_title("Correlación de Pearson · asalariados elegibles, 2025")
plt.tight_layout(); plt.show()
salary_corr = corr_table.loc["salario_mensual"].drop("salario_mensual")
strongest = salary_corr.abs().idxmax()
display(Markdown(f"La mayor asociación lineal absoluta con salario corresponde a **{strongest}** (r={salary_corr[strongest]:.3f}). Edad y antigüedad tienen r=**{corr_table.loc['edad','antiguedad']:.3f}**. Los resultados se refieren a las observaciones elegibles y no establecen causas."))
''')

md("""
## 4. Segmentación KMeans

Se comparan dos espacios de variables: perfil laboral (`edad`, `antiguedad`, `horas_semanales`) y sensibilidad con salario añadido. La primera define los perfiles principales; en ella el salario se usa **solo para caracterizarlos**. Se estandariza cada variable para evitar que su unidad domine la distancia. Para cada variante se prueban K=2–5 y tres semillas. El K con silhouette media más alta es seleccionado; si otro K está a menos de 0.01, se prefiere el menor. La silhouette de espacios distintos no se compara como prueba directa de superioridad.
""")
code(r'''
variants = {
    "perfil_laboral": ["edad", "antiguedad", "horas_semanales"],
    "con_salario": ["edad", "antiguedad", "horas_semanales", "salario_mensual"],
}
seeds = [42, 123, 2026]
evaluator = ClusteringEvaluator(featuresCol="features_scaled", predictionCol="prediction",
                                metricName="silhouette", distanceMeasure="squaredEuclidean")
experiments = []
variant_stages = {}
for variant, columns in variants.items():
    assembler = VectorAssembler(inputCols=columns, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled", withMean=True, withStd=True)
    scaler_model = scaler.fit(assembler.transform(train))
    scaled = scaler_model.transform(assembler.transform(train)).cache()
    scaled.count()
    variant_stages[variant] = (assembler, scaler_model)
    for k in [2, 3, 4, 5]:
        for seed in seeds:
            model = KMeans(featuresCol="features_scaled", predictionCol="prediction",
                           k=k, seed=seed, maxIter=100, tol=1e-4).fit(scaled)
            predicted = model.transform(scaled)
            silhouette = float(evaluator.evaluate(predicted))
            sizes = {int(r.prediction): int(r["count"]) for r in predicted.groupBy("prediction").count().collect()}
            cost = float(model.summary.trainingCost)
            experiments.append({"variante": variant, "K": k, "semilla": seed,
                                "silhouette": silhouette, "costo": cost,
                                "tamano_minimo": min(sizes.values()), "tamanos": sizes})
            print(variant, "K=", k, "semilla=", seed, "silhouette=", round(silhouette, 4), "tamaños=", sizes, flush=True)
    scaled.unpersist()
experiments_df = pd.DataFrame(experiments)
display(experiments_df)
''')
code(r'''
chosen = {}
for variant in variants:
    subset = experiments_df.loc[experiments_df.variante == variant]
    average = subset.groupby("K")["silhouette"].agg(["mean", "min", "max"])
    best_mean = average["mean"].max()
    chosen_k = min(int(k) for k in average.index if best_mean - average.loc[k, "mean"] <= 0.01)
    candidates = subset.loc[subset.K == chosen_k].sort_values(["silhouette", "semilla"], ascending=[False, True])
    winner = candidates.iloc[0]
    chosen[variant] = {"K": chosen_k, "semilla": int(winner.semilla), "silhouette": float(winner.silhouette)}
    display(Markdown(f"### {variant}: K={chosen_k}, semilla={int(winner.semilla)}"))
    display(average.round(4))
    display(Markdown(f"Silhouette media por semilla para K elegido: **{average.loc[chosen_k,'mean']:.4f}** (rango {average.loc[chosen_k,'min']:.4f}–{average.loc[chosen_k,'max']:.4f}). Tamaño mínimo del ajuste elegido: **{int(winner.tamano_minimo):,}** ({winner.tamano_minimo/train_n:.2%})."))
fig, ax = plt.subplots()
sns.lineplot(data=experiments_df.groupby(["variante", "K"], as_index=False)["silhouette"].mean(),
             x="K", y="silhouette", hue="variante", marker="o", ax=ax)
ax.set(title="Silhouette media por K y variante · 2025", ylabel="Silhouette euclidiana²", xticks=[2,3,4,5])
plt.tight_layout(); plt.show()
''')
code(r'''
variant = "perfil_laboral"
choice = chosen[variant]
assembler, scaler_model = variant_stages[variant]
scaled_for_choice = scaler_model.transform(assembler.transform(train))
best_model = KMeans(featuresCol="features_scaled", predictionCol="prediction",
                    k=choice["K"], seed=choice["semilla"], maxIter=100, tol=1e-4).fit(scaled_for_choice)
pipeline = PipelineModel(stages=[assembler, scaler_model, best_model])
assignments = pipeline.transform(train).cache()
assert assignments.count() == train_n
artifact_dir = ROOT / "artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)
pipeline.write().overwrite().save(str(artifact_dir / "kmeans_perfil_laboral"))
assignments.select(*KEY, "prediction").write.mode("overwrite").parquet(str(artifact_dir / "asignaciones_2025.parquet"))
profile_spark = assignments.groupBy("prediction").agg(
    F.count("*").alias("n"),
    *[F.avg(c).alias(f"media_{c}") for c in ["edad", "antiguedad", "horas_semanales", "salario_mensual"]],
    *[F.expr(f"percentile_approx({c}, 0.5, 10000)").alias(f"mediana_{c}")
      for c in ["edad", "antiguedad", "horas_semanales", "salario_mensual"]],
).orderBy("prediction")
profiles = profile_spark.toPandas()
profiles["porcentaje"] = 100 * profiles["n"] / train_n
display(profiles.round(2))
assert profiles.n.sum() == train_n
''')
code(r'''
composition = {}
for group in ["nivel_educativo", "categoria_ocupacional", "dominio"]:
    table = (assignments.groupBy("prediction", group).count()
             .withColumn("porcentaje_cluster", F.col("count") * 100 / F.sum("count").over(
                 __import__("pyspark").sql.Window.partitionBy("prediction")))
             .orderBy("prediction", F.desc("count")).toPandas())
    composition[group] = table
    display(Markdown(f"### Composición por {group}"))
    display(table.round(2))
profile_numeric = profiles.set_index("prediction")
overall = {c: float(summary.set_index("variable").loc[c, "media"]) for c in ["edad", "antiguedad", "horas_semanales"]}
names = {}
for row in profiles.itertuples():
    parts = []
    for label, field in [("edad", "edad"), ("antigüedad", "antiguedad"), ("jornada", "horas_semanales")]:
        ratio = getattr(row, f"media_{field}") / overall[field]
        parts.append(f"{label} {'alta' if ratio > 1.1 else 'baja' if ratio < 0.9 else 'intermedia'}")
    names[int(row.prediction)] = ", ".join(parts)
    display(Markdown(f"**Cluster {row.prediction}: {names[int(row.prediction)]}.** n={row.n:,} ({row.porcentaje:.1f}%), edad media {row.media_edad:.1f} años, antigüedad media {row.media_antiguedad:.1f} años, jornada media {row.media_horas_semanales:.1f} h/semana y salario mediano Q{row.mediana_salario_mensual:,.2f}."))
''')
code(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
profile_long = profiles.melt(id_vars="prediction", value_vars=["media_edad", "media_antiguedad", "media_horas_semanales"],
                              var_name="caracteristica", value_name="media")
sns.barplot(data=profile_long, x="prediction", y="media", hue="caracteristica", ax=axes[0])
axes[0].set(title="Perfil medio de los clusters · 2025", xlabel="Cluster", ylabel="Media (años u horas semanales)")
legend_handles, _ = axes[0].get_legend_handles_labels()
axes[0].get_legend().remove()
fig.legend(legend_handles, ["Edad (años)", "Antigüedad (años)", "Horas/semana"],
           loc="lower left", bbox_to_anchor=(0.06, -0.01), ncol=3, fontsize=8, frameon=False)
sns.barplot(data=profiles, x="prediction", y="mediana_salario_mensual", ax=axes[1], color="#d95f02")
axes[1].set(title="Salario mediano por cluster · 2025", xlabel="Cluster", ylabel="Quetzales mensuales")
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.show()
sample = (assignments.orderBy(F.rand(42))
          .select("edad", "antiguedad", "horas_semanales", "salario_mensual", "prediction")
          .limit(5000).toPandas())
assert len(sample) <= 5000
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=sample, x="edad", y="antiguedad", hue="prediction", alpha=0.45, s=18, ax=ax)
ax.set(title="Muestra visual de 5,000 registros · clusters 2025", xlabel="Edad (años)", ylabel="Antigüedad (años)")
plt.tight_layout(); plt.show()
display(Markdown("La variante con salario evalúa sensibilidad del agrupamiento. Su silhouette corresponde a un espacio de cuatro variables y no es directamente comparable con la de tres variables. Los perfiles principales se interpretan por edad, antigüedad y horas; sus diferencias salariales son descriptivas."))
''')

md("""
### Sensibilidad a incluir salario

Comparamos las asignaciones de las dos variantes para los mismos registros, sin confundir los identificadores de cluster (que pueden intercambiarse entre ajustes). La tabla muestra cuántas observaciones conservan su agrupación después de alinear las etiquetas. También se describe el salario de los grupos resultantes; esta comparación ayuda a decidir si añadirlo cambia el significado de los perfiles.
""")
code(r'''
salary_choice = chosen["con_salario"]
salary_assembler, salary_scaler = variant_stages["con_salario"]
salary_scaled_for_choice = salary_scaler.transform(salary_assembler.transform(train))
salary_model = KMeans(featuresCol="features_scaled", predictionCol="prediction",
                      k=salary_choice["K"], seed=salary_choice["semilla"],
                      maxIter=100, tol=1e-4).fit(salary_scaled_for_choice)
salary_pipeline = PipelineModel(stages=[salary_assembler, salary_scaler, salary_model])
salary_clusters = salary_pipeline.transform(train).cache()
salary_clusters.count()
salary_profile = (salary_clusters.groupBy("prediction")
                  .agg(F.count("*").alias("n"), F.avg("edad").alias("edad_media"),
                       F.avg("antiguedad").alias("antiguedad_media"),
                       F.avg("horas_semanales").alias("horas_medias"),
                       F.expr("percentile_approx(salario_mensual, 0.5, 10000)").alias("salario_mediano"))
                  .orderBy("prediction").toPandas())
display(salary_profile.round(2))
cross = (assignments.select(*KEY, F.col("prediction").alias("perfil"))
         .join(salary_clusters.select(*KEY, F.col("prediction").alias("con_salario")), KEY)
         .groupBy("perfil", "con_salario").count().toPandas())
cross_table = cross.pivot(index="perfil", columns="con_salario", values="count").fillna(0).astype(int)
display(cross_table)
assert cross_table.to_numpy().sum() == train_n
counts = cross_table.to_numpy()
agreement = max(counts[0,0] + counts[1,1], counts[0,1] + counts[1,0]) / train_n
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cross_table, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set(title="Cruce de asignaciones · 2025", xlabel="Cluster con salario", ylabel="Perfil laboral")
plt.tight_layout(); plt.show()
display(Markdown(f"Tras alinear las etiquetas, **{agreement:.1%}** de las observaciones permanecen en el mismo grupo y **{1-agreement:.1%}** cambian. La variante con salario produce grupos de tamaño {', '.join(f'{int(x):,}' for x in salary_profile.n)}. Dado que el objetivo es describir perfiles de edad, antigüedad y jornada sin definirlos por el ingreso, se conserva la variante laboral como principal. El salario se reporta aparte para comparar los perfiles; su inclusión modifica los grupos en la proporción observada arriba."))
_ = salary_clusters.unpersist()
_ = assignments.unpersist()
for frame in raws.values():
    frame.unpersist()
for period, frame in prepared.items():
    if period != "2026T1":
        frame.unpersist()
del best_model, pipeline, salary_model, salary_pipeline, variant_stages
''')

md("""
## 5. Pipeline de regresión lineal

La selección de hiperparámetros respeta el orden temporal indicado en la guía: **2025T1–T3** forman el conjunto de entrenamiento y **2025T4** es la validación. El primer trimestre de 2026 no interviene en esta selección. El referente es una predicción constante igual al salario promedio del conjunto de entrenamiento; por tanto, tampoco usa información de validación.

Los tres predictores categóricos se ajustan con `StringIndexer` y `OneHotEncoder`; `handleInvalid="keep"` permite aplicar el pipeline a categorías nuevas sin volver a ajustarlo. Las variables numéricas son edad, antigüedad y horas semanales. `LinearRegression(standardization=True)` estandariza internamente para la penalización. Se conserva el salario original en quetzales, sin recortes ni transformaciones.
""")
code(r'''
categorical_predictors = ["nivel_educativo", "categoria_ocupacional", "dominio"]
numeric_predictors = ["edad", "antiguedad", "horas_semanales"]
supervised_predictors = numeric_predictors + categorical_predictors

development_2025 = train.filter(F.col("periodo_archivo").isin("2025T1", "2025T2", "2025T3")).cache()
validation_2025 = train.filter(F.col("periodo_archivo") == "2025T4").cache()
development_n, validation_n = development_2025.count(), validation_2025.count()
assert development_n + validation_n == train_n
assert set(development_2025.select("periodo_archivo").distinct().toPandas()["periodo_archivo"]) == {"2025T1", "2025T2", "2025T3"}
assert set(validation_2025.select("periodo_archivo").distinct().toPandas()["periodo_archivo"]) == {"2025T4"}
print("Entrenamiento 2025T1-T3:", development_n, "| Validación 2025T4:", validation_n,
      "| Prueba reservada 2026T1:", test_n)

def preprocessing_stages():
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep",
                              stringOrderType="alphabetAsc") for c in categorical_predictors]
    encoded = [f"{c}_ohe" for c in categorical_predictors]
    encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in categorical_predictors],
                            outputCols=encoded, handleInvalid="keep", dropLast=True)
    assembler = VectorAssembler(inputCols=numeric_predictors + encoded, outputCol="features",
                                handleInvalid="error")
    return indexers + [encoder, assembler]

def regression_metrics(predictions):
    errors = predictions.select(
        (F.col("salario_mensual") - F.col("prediction")).alias("residuo"))
    row = errors.agg(F.avg(F.abs("residuo")).alias("MAE"),
                     F.sqrt(F.avg(F.pow("residuo", 2))).alias("RMSE")).first()
    r2 = RegressionEvaluator(labelCol="salario_mensual", predictionCol="prediction",
                             metricName="r2").evaluate(predictions)
    return {"MAE": float(row.MAE), "RMSE": float(row.RMSE), "R2": float(r2)}

def constant_baseline(fit_frame, evaluation_frame):
    value = float(fit_frame.agg(F.avg("salario_mensual").alias("media")).first().media)
    predictions = evaluation_frame.withColumn("prediction", F.lit(value))
    return value, regression_metrics(predictions)

baseline_development_value, baseline_validation_metrics = constant_baseline(
    development_2025, validation_2025)
display(pd.DataFrame([{"modelo": "Referencia: media de entrenamiento",
                       "configuracion": f"predicción constante Q{baseline_development_value:,.2f}",
                       **baseline_validation_metrics}]).round(3))
''')
code(r'''
linear_configs = [
    {"nombre": "sin_regularizacion", "regParam": 0.0, "elasticNetParam": 0.0},
    {"nombre": "ridge_1", "regParam": 1.0, "elasticNetParam": 0.0},
    {"nombre": "ridge_10", "regParam": 10.0, "elasticNetParam": 0.0},
    {"nombre": "elastic_net_10", "regParam": 10.0, "elasticNetParam": 0.5},
]
linear_results, linear_models = [], {}
for config in linear_configs:
    estimator = LinearRegression(
        featuresCol="features", labelCol="salario_mensual", predictionCol="prediction",
        regParam=config["regParam"], elasticNetParam=config["elasticNetParam"],
        standardization=True, maxIter=200, tol=1e-6)
    model = Pipeline(stages=preprocessing_stages() + [estimator]).fit(development_2025)
    predictions = model.transform(validation_2025).cache()
    assert predictions.count() == validation_n
    metrics = regression_metrics(predictions)
    linear_results.append({"modelo": "Regresión lineal", "configuracion": config["nombre"],
                           "regParam": config["regParam"],
                           "elasticNetParam": config["elasticNetParam"], **metrics})
    linear_models[config["nombre"]] = model
    predictions.unpersist()
    print(config["nombre"], metrics, flush=True)

linear_results_df = pd.DataFrame(linear_results).sort_values("RMSE").reset_index(drop=True)
display(linear_results_df.round(3))
best_linear_row = linear_results_df.iloc[0]
best_linear_config = next(c for c in linear_configs if c["nombre"] == best_linear_row["configuracion"])
best_linear_development = linear_models[best_linear_config["nombre"]]
models_dir = ROOT / "artifacts" / "models"
models_dir.mkdir(parents=True, exist_ok=True)
best_linear_development.write().overwrite().save(str(models_dir / "linear_validacion_2025"))
improvement_linear = 100 * (baseline_validation_metrics["RMSE"] - best_linear_row["RMSE"]) / baseline_validation_metrics["RMSE"]
display(Markdown(
    f"La configuración seleccionada es **{best_linear_config['nombre']}** "
    f"(`regParam={best_linear_config['regParam']}`, `elasticNetParam={best_linear_config['elasticNetParam']}`), "
    f"con RMSE de validación **Q{best_linear_row['RMSE']:,.2f}**, MAE **Q{best_linear_row['MAE']:,.2f}** "
    f"y R² **{best_linear_row['R2']:.3f}**. Su RMSE es {abs(improvement_linear):.1f}% "
    f"{'menor' if improvement_linear >= 0 else 'mayor'} que el referente constante. "
    "La configuración se eligió únicamente por el menor RMSE de 2025T4."))
del linear_models, best_linear_development
''')

md("""
## 6. Pipeline de Random Forest

Se reutilizan exactamente los mismos conjuntos de entrenamiento y validación. Aunque los árboles no requieren estandarización, el pipeline incluye la codificación categórica exigida. Se comparan dos combinaciones de cantidad de árboles y profundidad, ambas con semilla fija. Todos los `StringIndexer`, el `OneHotEncoder` y el estimador se ajustan solo con 2025T1–T3.
""")
code(r'''
rf_configs = [
    {"nombre": "rf_40_d7", "numTrees": 40, "maxDepth": 7},
    {"nombre": "rf_80_d10", "numTrees": 80, "maxDepth": 10},
]
rf_results, rf_models = [], {}
for config in rf_configs:
    estimator = RandomForestRegressor(
        featuresCol="features", labelCol="salario_mensual", predictionCol="prediction",
        numTrees=config["numTrees"], maxDepth=config["maxDepth"], seed=42,
        minInstancesPerNode=5, subsamplingRate=0.8, featureSubsetStrategy="auto")
    model = Pipeline(stages=preprocessing_stages() + [estimator]).fit(development_2025)
    predictions = model.transform(validation_2025).cache()
    assert predictions.count() == validation_n
    metrics = regression_metrics(predictions)
    rf_results.append({"modelo": "Random Forest", "configuracion": config["nombre"],
                       "numTrees": config["numTrees"], "maxDepth": config["maxDepth"],
                       "semilla": 42, **metrics})
    rf_models[config["nombre"]] = model
    predictions.unpersist()
    print(config["nombre"], metrics, flush=True)

rf_results_df = pd.DataFrame(rf_results).sort_values("RMSE").reset_index(drop=True)
display(rf_results_df.round(3))
best_rf_row = rf_results_df.iloc[0]
best_rf_config = next(c for c in rf_configs if c["nombre"] == best_rf_row["configuracion"])
best_rf_development = rf_models[best_rf_config["nombre"]]
best_rf_development.write().overwrite().save(str(models_dir / "random_forest_validacion_2025"))
improvement_rf = 100 * (baseline_validation_metrics["RMSE"] - best_rf_row["RMSE"]) / baseline_validation_metrics["RMSE"]
display(Markdown(
    f"La configuración seleccionada es **{best_rf_config['nombre']}** "
    f"({best_rf_config['numTrees']} árboles, profundidad máxima {best_rf_config['maxDepth']}, semilla 42), "
    f"con RMSE **Q{best_rf_row['RMSE']:,.2f}**, MAE **Q{best_rf_row['MAE']:,.2f}** y "
    f"R² **{best_rf_row['R2']:.3f}**. Su RMSE es {abs(improvement_rf):.1f}% "
    f"{'menor' if improvement_rf >= 0 else 'mayor'} que el referente constante."))
del rf_models, best_rf_development
''')
code(r'''
validation_comparison = pd.DataFrame([
    {"modelo": "Referencia", "configuracion": f"media Q{baseline_development_value:,.2f}",
     **baseline_validation_metrics},
    {"modelo": "Regresión lineal", "configuracion": best_linear_config["nombre"],
     "MAE": float(best_linear_row.MAE), "RMSE": float(best_linear_row.RMSE), "R2": float(best_linear_row.R2)},
    {"modelo": "Random Forest", "configuracion": best_rf_config["nombre"],
     "MAE": float(best_rf_row.MAE), "RMSE": float(best_rf_row.RMSE), "R2": float(best_rf_row.R2)},
]).sort_values("RMSE")
display(validation_comparison.round(3))
best_validation = validation_comparison.loc[validation_comparison.modelo != "Referencia"].sort_values("RMSE").iloc[0]
other_validation = validation_comparison.loc[
    (validation_comparison.modelo != "Referencia") &
    (validation_comparison.modelo != best_validation.modelo)].iloc[0]
explanation = ("puede representar relaciones no lineales e interacciones entre edad, antigüedad, jornada y categorías"
               if best_validation.modelo == "Random Forest" else
               "su estructura aditiva parece generalizar mejor en este corte y evita parte de la varianza del bosque")
display(Markdown(
    f"En 2025T4, **{best_validation.modelo}** obtiene el menor RMSE de los dos algoritmos "
    f"(Q{best_validation.RMSE:,.2f} frente a Q{other_validation.RMSE:,.2f}). Esto es compatible con que {explanation}. "
    "La comparación es predictiva y no implica que los coeficientes, importancias o asociaciones sean causales."))
''')

md("""
## 7. Entrenamiento final y evaluación en 2026

Después de congelar una configuración por algoritmo, cada pipeline se vuelve a ajustar con **los cuatro trimestres de 2025**. Solo entonces se transforma 2026T1. Ambos modelos se evalúan sobre las mismas observaciones elegibles y con MAE, RMSE y R². El referente final se recalcula con la media de 2025, sin mirar los salarios de 2026.
""")
code(r'''
selected_linear_estimator = LinearRegression(
    featuresCol="features", labelCol="salario_mensual", predictionCol="prediction",
    regParam=best_linear_config["regParam"], elasticNetParam=best_linear_config["elasticNetParam"],
    standardization=True, maxIter=200, tol=1e-6)
selected_rf_estimator = RandomForestRegressor(
    featuresCol="features", labelCol="salario_mensual", predictionCol="prediction",
    numTrees=best_rf_config["numTrees"], maxDepth=best_rf_config["maxDepth"], seed=42,
    minInstancesPerNode=5, subsamplingRate=0.8, featureSubsetStrategy="auto")

linear_final_model = Pipeline(stages=preprocessing_stages() + [selected_linear_estimator]).fit(train)
rf_final_model = Pipeline(stages=preprocessing_stages() + [selected_rf_estimator]).fit(train)
linear_final_model.write().overwrite().save(str(models_dir / "linear_final_2025"))
rf_final_model.write().overwrite().save(str(models_dir / "random_forest_final_2025"))

linear_test_predictions = linear_final_model.transform(test_2026).cache()
rf_test_predictions = rf_final_model.transform(test_2026).cache()
assert linear_test_predictions.count() == test_n == rf_test_predictions.count()
baseline_final_value, baseline_test_metrics = constant_baseline(train, test_2026)
linear_test_metrics = regression_metrics(linear_test_predictions)
rf_test_metrics = regression_metrics(rf_test_predictions)
test_comparison = pd.DataFrame([
    {"modelo": "Referencia", "configuracion": f"media 2025 = Q{baseline_final_value:,.2f}", **baseline_test_metrics},
    {"modelo": "Regresión lineal", "configuracion": best_linear_config["nombre"], **linear_test_metrics},
    {"modelo": "Random Forest", "configuracion": best_rf_config["nombre"], **rf_test_metrics},
]).sort_values("RMSE").reset_index(drop=True)
display(test_comparison.round(3))

best_test = test_comparison.loc[test_comparison.modelo != "Referencia"].sort_values("RMSE").iloc[0]
baseline_test_row = test_comparison.loc[test_comparison.modelo == "Referencia"].iloc[0]
test_gain = 100 * (baseline_test_row.RMSE - best_test.RMSE) / baseline_test_row.RMSE
display(Markdown(
    f"En la prueba final de **{test_n:,}** registros de 2026T1, **{best_test.modelo}** logra el menor RMSE "
    f"entre los dos algoritmos: **Q{best_test.RMSE:,.2f}**, con MAE **Q{best_test.MAE:,.2f}** y "
    f"R² **{best_test.R2:.3f}**. El cambio de RMSE frente al referente es {abs(test_gain):.1f}% "
    f"{'menor' if test_gain >= 0 else 'mayor'}. Esta es la única evaluación usada para reportar desempeño final."))

metrics_artifact = {
    "particiones": {"entrenamiento_2025T1_T3": development_n, "validacion_2025T4": validation_n,
                    "entrenamiento_final_2025": train_n, "prueba_2026T1": test_n},
    "seleccion": {"regresion_lineal": best_linear_config, "random_forest": {**best_rf_config, "seed": 42}},
    "validacion_2025T4": validation_comparison.to_dict(orient="records"),
    "prueba_2026T1": test_comparison.to_dict(orient="records"),
    "residuo": "salario_mensual - prediction; positivo = subestimacion",
}
_ = (ROOT / "artifacts" / "supervised_metrics.json").write_text(
    json.dumps(metrics_artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
''')

md("""
## 8. Visualización y análisis de errores

Se define `residuo = salario_mensual - prediction`: un valor positivo indica subestimación y uno negativo, sobreestimación. Los gráficos de ambos modelos usan la **misma muestra determinística de hasta 5,000 registros** de 2026T1. Las tablas por educación, dominio y percentil se calculan con la prueba completa, no con la muestra.
""")
code(r'''
sample_columns = KEY + supervised_predictors + ["salario_mensual"]
sample_order = F.sha2(F.concat_ws("||", *[F.coalesce(F.col(c).cast("string"), F.lit("<NULL>"))
                                           for c in sample_columns]), 256)
visual_sample = test_2026.orderBy(sample_order).limit(5000).cache()
visual_n = visual_sample.count()
linear_visual = (linear_final_model.transform(visual_sample)
                 .select("salario_mensual", "prediction").toPandas())
rf_visual = (rf_final_model.transform(visual_sample)
             .select("salario_mensual", "prediction").toPandas())
assert len(linear_visual) == visual_n == len(rf_visual) <= 5000
for frame in [linear_visual, rf_visual]:
    frame["residuo"] = frame["salario_mensual"] - frame["prediction"]

joint_min = min(0.0, linear_visual[["salario_mensual", "prediction"]].to_numpy().min(),
                rf_visual[["salario_mensual", "prediction"]].to_numpy().min())
joint_max = max(linear_visual[["salario_mensual", "prediction"]].to_numpy().max(),
                rf_visual[["salario_mensual", "prediction"]].to_numpy().max())
fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), sharex=True, sharey=True)
for ax, frame, title in [(axes[0], linear_visual, "Regresión lineal"),
                         (axes[1], rf_visual, "Random Forest")]:
    ax.scatter(frame["salario_mensual"], frame["prediction"], alpha=0.25, s=14, color="#2374ab")
    ax.plot([joint_min, joint_max], [joint_min, joint_max], "--", color="#d95f02", linewidth=1.5)
    ax.set(title=title, xlabel="Salario real (Q)", ylabel="Salario predicho (Q)",
           xlim=(joint_min, joint_max), ylim=(joint_min, joint_max))
fig.suptitle(f"Salario real frente a predicho · misma muestra 2026T1 (n={visual_n:,})")
plt.tight_layout(); plt.show()
''')
code(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), sharey=True)
for ax, frame, title in [(axes[0], linear_visual, "Regresión lineal"),
                         (axes[1], rf_visual, "Random Forest")]:
    ax.scatter(frame["prediction"], frame["residuo"], alpha=0.25, s=14, color="#2a9d8f")
    ax.axhline(0, linestyle="--", color="#d95f02", linewidth=1.5)
    ax.set(title=title, xlabel="Salario predicho (Q)", ylabel="Residuo: real - predicho (Q)")
fig.suptitle(f"Residuos frente a predicción · misma muestra 2026T1 (n={visual_n:,})")
plt.tight_layout(); plt.show()
''')
code(r'''
def grouped_errors(predictions, group, model_name):
    return (predictions.withColumn("residuo", F.col("salario_mensual") - F.col("prediction"))
            .groupBy(group)
            .agg(F.count("*").alias("n"),
                 F.avg(F.abs("residuo")).alias("MAE"),
                 F.avg("residuo").alias("error_medio"))
            .withColumn("modelo", F.lit(model_name))
            .select("modelo", group, "n", "MAE", "error_medio")
            .orderBy("modelo", group).toPandas())

group_error_tables = {}
for group in ["nivel_educativo", "dominio"]:
    table = pd.concat([
        grouped_errors(linear_test_predictions, group, "Regresión lineal"),
        grouped_errors(rf_test_predictions, group, "Random Forest")
    ], ignore_index=True)
    group_error_tables[group] = table
    assert all(table.groupby("modelo")["n"].sum() == test_n)
    display(Markdown(f"### Error por {group.replace('_', ' ')} · prueba completa 2026T1"))
    display(table.round(2))
''')
code(r'''
percentile_values = test_2026.agg(
    F.expr("percentile_approx(salario_mensual, array(0.5, 0.75, 0.9, 0.95), 10000)").alias("p")
).first().p
p50, p75, p90, p95 = [float(x) for x in percentile_values]

def add_salary_band(frame):
    return frame.withColumn(
        "percentil_salario",
        F.when(F.col("salario_mensual") <= p50, "P00-P50")
         .when(F.col("salario_mensual") <= p75, "P50-P75")
         .when(F.col("salario_mensual") <= p90, "P75-P90")
         .when(F.col("salario_mensual") <= p95, "P90-P95")
         .otherwise("P95-P100"))

percentile_errors = pd.concat([
    grouped_errors(add_salary_band(linear_test_predictions), "percentil_salario", "Regresión lineal"),
    grouped_errors(add_salary_band(rf_test_predictions), "percentil_salario", "Random Forest")
], ignore_index=True)
assert all(percentile_errors.groupby("modelo")["n"].sum() == test_n)
display(pd.DataFrame({"percentil": ["P50", "P75", "P90", "P95"],
                      "salario_mensual_Q": [p50, p75, p90, p95]}).round(2))
display(percentile_errors.round(2))

test_salary_summary = stats(test_2026, "salario_mensual")
tail = percentile_errors.loc[percentile_errors.percentil_salario == "P95-P100"].set_index("modelo")
def bias_text(value):
    if abs(value) < 1e-9:
        return "sin sesgo medio"
    return "subestimación" if value > 0 else "sobreestimación"
display(Markdown(
    f"En 2026T1 la media salarial es **Q{test_salary_summary['media']:,.2f}** y la mediana "
    f"**Q{test_salary_summary['mediana']:,.2f}**; su diferencia y el máximo de "
    f"**Q{test_salary_summary['maximo']:,.2f}** confirman una cola derecha. Por encima de P95 "
    f"(Q{p95:,.2f}), la regresión lineal tiene error medio **Q{tail.loc['Regresión lineal','error_medio']:,.2f}** "
    f"({bias_text(tail.loc['Regresión lineal','error_medio'])}) y Random Forest **Q{tail.loc['Random Forest','error_medio']:,.2f}** "
    f"({bias_text(tail.loc['Random Forest','error_medio'])}). El crecimiento del MAE en los percentiles altos muestra "
    "la influencia de la cola salarial: con pérdida cuadrática, pocos errores grandes pueden dominar el RMSE. "
    "Los extremos se conservaron tal como exige la actividad."))
''')

md("""
## Discusión final

Los resultados deben leerse como desempeño sobre **registros elegibles no ponderados**, no como una estimación oficial de salarios ni como una recomendación salarial. Las diferencias por educación, ocupación, dominio o cluster son asociaciones descriptivas; no controlan todas las variables relevantes y no prueban causalidad. La rotación longitudinal significa que las filas tampoco equivalen a personas únicas.
""")
code(r'''
validation_winner = validation_comparison.loc[validation_comparison.modelo != "Referencia"].sort_values("RMSE").iloc[0]
test_winner = test_comparison.loc[test_comparison.modelo != "Referencia"].sort_values("RMSE").iloc[0]
same_winner = validation_winner.modelo == test_winner.modelo
display(Markdown(
    f"La exploración encontró un salario asimétrico y perfiles laborales diferenciados por edad, antigüedad y jornada. "
    f"En validación ganó **{validation_winner.modelo}** (RMSE Q{validation_winner.RMSE:,.2f}); en la prueba 2026 ganó "
    f"**{test_winner.modelo}** (RMSE Q{test_winner.RMSE:,.2f}). "
    f"{'La coincidencia del algoritmo ganador entre validación y prueba aporta estabilidad a la selección.' if same_winner else 'El cambio de algoritmo ganador muestra que la comparación es sensible al período y debe reportarse sin escoger de nuevo con 2026.'} "
    f"Aun para el mejor modelo final, R²={test_winner.R2:.3f}: las seis variables observadas explican solo parte de la "
    "variación salarial. Los errores por grupo y percentil muestran que un promedio global puede ocultar diferencias "
    "importantes, especialmente en la cola alta. `FACTOR` se conserva para un análisis poblacional posterior que respete "
    "el diseño muestral, pero no se utilizó como predictor ni como ponderador en esta comparación obligatoria."))
''')

md("""
## Validaciones de integridad y reproducibilidad

Las comprobaciones siguientes verifican las particiones temporales, la cobertura de los experimentos, el tamaño común de prueba, la muestra gráfica y la persistencia de modelos y métricas. Una ejecución limpia debe llegar a esta celda sin depender de estado manual previo.
""")
code(r'''
assert train_n > 0 and test_n > 0
assert all(len(experiments_df.loc[experiments_df.variante == v]) == 12 for v in variants)
assert len(linear_results_df) >= 2 and len(rf_results_df) >= 2
assert set(train.select("periodo_archivo").distinct().toPandas()["periodo_archivo"]) == set(periodos[:4])
assert set(test_2026.select("periodo_archivo").distinct().toPandas()["periodo_archivo"]) == {"2026T1"}
assert linear_test_predictions.select(*supervised_predictors, "salario_mensual").count() == test_n
assert rf_test_predictions.select(*supervised_predictors, "salario_mensual").count() == test_n
assert visual_n <= 5000
for path in [models_dir / "linear_validacion_2025", models_dir / "random_forest_validacion_2025",
             models_dir / "linear_final_2025", models_dir / "random_forest_final_2025",
             ROOT / "artifacts" / "supervised_metrics.json"]:
    assert path.exists(), path
print("Validaciones finales superadas. Registros 2025:", train_n,
      "| registros 2026T1:", test_n, "| muestra visual común:", visual_n)
spark.stop()
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
path = ROOT / "Laboratorio7_EDA_Segmentacion.ipynb"
nbf.write(nb, path)
print(f"Escrito {path} con {len(cells)} celdas")

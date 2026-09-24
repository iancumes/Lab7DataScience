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
# Laboratorio 7 — ENEIC: análisis exploratorio y segmentación

**CC3066 Data Science · Avance de los incisos 1–4 · 24 de septiembre de 2026**

Este cuaderno estudia observaciones de personas asalariadas de 15 años o más con salario mensual positivo registrado. Los cuatro archivos de 2025 constituyen el análisis; 2026T1 se prepara y conserva para la entrega final. Las cifras son **no ponderadas** y no representan estimaciones oficiales de Guatemala. El mismo individuo puede aportar observaciones en períodos distintos.

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
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Correlation
from scripts.analysis import (ROOT, SOURCE_COLUMNS, KEY, load_excel, dictionary_codes,
                              audit_duplicates, missingness, prepare, stats, group_salary)

spark = (SparkSession.builder.master("local[4]").appName("Lab7-ENEIC-EDA-KMeans")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.driver.memory", "4g")
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
fitted = {}
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
            fitted[(variant, k, seed)] = model
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
best_model = fitted[(variant, choice["K"], choice["semilla"])]
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
salary_model = fitted[("con_salario", salary_choice["K"], salary_choice["semilla"])]
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
''')

md("""
## Síntesis del avance

El notebook deja trazabilidad del origen, esquema, pérdida por filtros, distribuciones, correlaciones y selección de K. Las observaciones longitudinales no se convierten en individuos únicos. Los patrones de salario y clusters son asociaciones de registros elegibles, sin lectura causal o poblacional. La preparación 2026 queda almacenada para los incisos 5–8, pero no participa en resultados de este avance.
""")
code(r'''
assert train_n > 0 and test_n > 0
assert all(len(experiments_df.loc[experiments_df.variante == v]) == 12 for v in variants)
assert set(train.select("periodo_archivo").distinct().toPandas()["periodo_archivo"]) == set(periodos[:4])
assert set(test_2026.select("periodo_archivo").distinct().toPandas()["periodo_archivo"]) == {"2026T1"}
print("Validaciones finales superadas. Registros 2025:", train_n, "; registros 2026T1:", test_n)
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
path = ROOT / "Laboratorio7_EDA_Segmentacion.ipynb"
nbf.write(nb, path)
print(f"Escrito {path} con {len(cells)} celdas")

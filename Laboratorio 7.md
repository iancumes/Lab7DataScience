Universidad del Valle de Guatemala

Facultad de Ingeniería

CC3066 – Data Science

## INSTRUCCIONES:

En este laboratorio aplicará técnicas de análisis exploratorio, aprendizaje no supervisado y regresión utilizando Python y Spark 3.5, mediante la API de DataFrames pyspark.ml.

Trabajará con las bases de Personas de la Encuesta Nacional de Empleo e Ingresos Continua —ENEIC— del Instituto Nacional de Estadística de Guatemala. Utilizará los cuatro trimestres de 2025 para desarrollar los modelos (train) y el primer trimestre de 2026 para la evaluación final (test).

El trabajo debe realizarse de forma grupal. Cada integrante deberá participar en la implementación y en la interpretación de resultados. Utilice un repositorio Git para registrar las contribuciones del equipo.

Presente el procedimiento en un notebook, combinando código, resultados, gráficos y explicaciones en celdas Markdown. Mostrar una tabla o una gráfica sin interpretarla no constituye una respuesta completa.

## CONTEXTO E IMPORTANCIA DEL ANÁLISIS

El estudio de los salarios permite identificar diferencias entre grupos de trabajadores, comprender su relación con la educación y las condiciones laborales y generar evidencia para orientar programas de capacitación y análisis del mercado de trabajo.

En este ejercicio, su equipo actuará como una unidad de análisis que debe responder dos preguntas:

- \- ¿Qué perfiles de trabajadores asalariados pueden identificarse según variables como su edad, antigüedad, jornada habitual, etc.?

- \- ¿Qué tan bien puede estimarse el salario mensual de una persona asalariada utilizando características personales y laborales observadas?

Los resultados permitirán comparar perfiles y evaluar la capacidad predictiva de dos modelos. Las asociaciones encontradas no demuestran relaciones causales ni deben interpretarse como recomendaciones sobre cuánto debería ganar una persona.

## Ambiente y recomendaciones generales:

Utilice el ambiente del curso con Python y Spark 3.5.x. Puede trabajar en Jupyter sobre el docker compartido previamente, o bien en cualquier otro entorno/plataforma, siempre que el entorno utilizado corresponda a la versión requerida.


## Universidad del Valle de Guatemala Facultad de Ingeniería

CC3066 – Data Science

Spark no incorpora un lector nativo de Excel. Para este laboratorio se permite leer cada archivo con pandas y/o librerías nativas como openpyxl, convertirlo a un DataFrame de Spark con tipos explícitos y guardarlo en Parquet (recomendado). Procese los archivos individualmente para controlar el consumo de memoria. Siéntanse en la libertad de usar funciones en Spark como las de persist o caché si considera que puede hacer más eficiente su trabajo.

A partir de esa conversión, realice la preparación analítica y el aprendizaje automático con Spark (MLlib). No utilice scikit-learn para entrenar los modelos. Para graficar, transfiera a pandas únicamente tablas agregadas o muestras de hasta 5,000 – 10,000 registros. Las métricas y las estadísticas deben calcularse sobre el conjunto completo correspondiente, no sobre la muestra usada para dibujar.

Organice el notebook de manera que pueda ejecutarse de principio a fin, sin depender de variables creadas manualmente en ejecuciones anteriores.

## EJERCICIOS

## DESCRIPCIÓN DE LOS DATOS

La ENEIC recopila información sobre personas, educación, empleo e ingresos. En las bases utilizadas, cada registro corresponde a una persona observada en un período. La encuesta tiene un diseño longitudinal con rotación. Por ello, la unión de varios trimestres contiene observaciones de personas que pueden aparecer más de una vez. El número de filas acumulado no equivale necesariamente al número de personas distintas.

Utilice exclusivamente las bases de Personas, cada base de datos viene acompañada por un diccionario de datos correspondiente.

|   | Período del archivo Registros originales | Columnas originales | Uso |
| --- | --- | --- | --- |
| I de 2025 | 51,588 | 270 | Desarrollo y entrenamiento |
| II de 2025 | 51,167 | 270 | Desarrollo y entrenamiento |
| III de 2025 | 51,583 | 270 | Desarrollo y entrenamiento |
| IV de 2025 | 49,338 | 302 | Validación y posterior entrenamiento final |
| I de 2026 | 49,843 | 270 | Prueba final |


Universidad del Valle de Guatemala

Facultad de Ingeniería

Departamento de Ciencias de la Computación

CC3066 – Data Science

NOTA: Estos conteos corresponden a los archivos proporcionados antes de aplicar filtros. FUENTE: https://www.ine.gob.gt/encuesta-nacional-de-empleo-e-ingresos/ [URL 🔗](https://www.ine.gob.gt/encuesta-nacional-de-empleo-e-ingresos/)

## IDENTIFICACIÓN DEL PERÍODO

Conserve la columna original TRIMESTRE, pero no la utilice directamente como trimestre calendario. En los archivos proporcionados se observaron los siguientes valores:

## Archivo Valores originales de TRIMESTRE

I de 2025 2 II de 2025 3 y, en 175 registros, 2 III de 2025 4 IV de 2025 5

I de 2026 6

Cree las columnas periodo_archivo, anio_archivo y trimestre_calendario a partir del archivo de

procedencia. Por ejemplo, el archivo de I de 2025 deberá identificarse como 2025T1, año 2025 y trimestre calendario 1.

Esta asignación identifica el corte publicado al que pertenece el archivo; no modifica ni pretende corregir la respuesta original. Conserve también archivo_origen para asegurar trazabilidad.

No reste uno automáticamente a TRIMESTRE, pues esa operación no resuelve todos los registros observados.

## POBLACIÓN Y VARIABLE OBJETIVO

El análisis principal se limitará a personas:

- De 15 años o más.

- Identificadas como ocupadas: OCUPADOS = 1.

- Asalariadas según P05C16, con códigos 1, 2, 3 o 4.

- Con un valor numérico, finito y estrictamente positivo en P05D01.

Las categorías de P05C16 incluidas son: 1 Empleado de gobierno; 2 empleado de empresa privada; 3 empleado jornalero o peón; y 4→ servicio doméstico.


Universidad del Valle de Guatemala

Facultad de Ingeniería

CC3066 – Data Science

Defina salario_mensual a partir de P05D01: sueldo o salario mensual sin descuentos de la ocupación principal, expresado en quetzales.

El objetivo no incluye automáticamente otros empleos, remesas, prestaciones o ingresos independientes. No sustituya esta variable por YLAB_PUBLI ni sume otros componentes.

Al excluir salarios nulos o no positivos, los resultados se referirán específicamente a asalariados con salario positivo registrado.

## VARIABLES QUE UTILIZARÁ

| Variable original | Nombre analítico | Uso |
| --- | --- | --- |
| P05D01 | salario_mensual | Variable objetivo |
| P02A03 | edad | Predictor numérico y clustering |
| P05C07A | antiguedad_anios | Construcción de antigüedad |
| P05C07B | antiguedad_meses | Construcción de antigüedad |
| P05H01A | horas_semanales | Horas habituales de la ocupación principal; predictor y clustering |
| P03A03A | nivel_educativo | Predictor categórico |
| P05C16 | categoria_ocupacional | Filtro y predictor categórico |
| DOMINIO | dominio | Predictor categórico |
| OCUPADOS | ocupado | Filtro |
| NUM_HOGAR, NUM_PERSONA | Conservar nombres | Auditoría de registros |
| FACTOR |   | Conservar nombre Documentación del diseño muestral |
| ANIO, TRIMESTRE | Conservar nombres | Auditoría de la fuente |

Construya: Antigüedad (en años) = antigüedad_anios + (antigüedad_meses / 12)

La antigüedad quedará expresada en años.

Para los modelos supervisados utilizará exactamente seis predictores: edad, antiguedad,

horas_semanales, nivel_educativo, categoria_ocupacional y dominio.

No incorpore identificadores, el factor de expansión, otros montos de ingresos, el salario por hora derivado del objetivo ni la etiqueta de cluster como predictores.

## CRITERIOS COMUNES DE PREPARACIÓN


Convierta los códigos a una representación consistente antes de unir los archivos. Un mismo código puede llegar como número o como texto.

Para las variables numéricas utilizadas, conserve registros con:

- Edad finita y mayor o igual a 15.

- Antigüedad en años no negativa.

- Componente de meses entero entre 0 y 11.

- Antigüedad calculada menor o igual a la edad.

- Horas habituales mayores que cero y menores o iguales a 168 por semana.

Excluya y contabilice los registros que no permitan evaluar esos criterios. No impute el salario.

Para las variables categóricas, valide los códigos contra el diccionario. Represente los

valores ausentes o no reconocidos como DESCONOCIDO; no los convierta arbitrariamente a cero. En particular, el código educativo 0 significa “ninguno” y no es un faltante.

No elimine automáticamente salarios altos o bajos por ser extremos. Manténgalos en la evaluación principal y discuta su influencia. No aplique recortes por percentiles ni transformaciones del objetivo en la comparación obligatoria.

Para mantener un alcance uniforme, el clustering, los modelos y las métricas principales serán no ponderados. Los resultados describirán los registros analizados; no deben presentarse como estimaciones oficiales de la población guatemalteca. Conserve FACTOR y explique para qué se utilizaría en un análisis poblacional.

## ANÁLISIS EXPLORATORIO AVANZADO Y SEGMENTACIÓN (25 pts)

A continuación, responda las siguientes preguntas:

## 1. Carga, armonización y calidad de datos — 5 puntos

Cargue los archivos, identifique su procedencia, seleccione las columnas requeridas y homologue sus tipos. Una los cuatro archivos de 2025 mediante unionByName.

## Muestre:

- Esquema y cinco registros de las columnas seleccionadas.

- Número de registros por archivo antes y después de los filtros.


- Cantidad y porcentaje de faltantes por variable seleccionada, antes de aplicar los filtros.

- Número de registros excluidos en cada paso, utilizando siempre el mismo orden de filtrado.

- Verificación de unicidad de la combinación periodo_archivo, NUM_HOGAR y NUM_PERSONA.

Si encuentra claves duplicadas, investigue si son repeticiones exactas o registros en conflicto. No utilice dropDuplicates() para ocultar el problema.

## Responda:

- ¿Por qué IV de 2025 no puede apilarse por posición de columnas con los otros archivos?

- ¿Qué diferencia existe entre un dato ausente porque la pregunta no corresponde y una respuesta no registrada?

- ¿Por qué una persona observada en dos períodos no debe eliminarse como duplicado del conjunto longitudinal?

- ¿Por qué el número de registros de la base filtrada no representa a todos los trabajadores del país?

Guarde el conjunto preparado de 2025 y el de 2026 por separado en Parquet.

## 2. Estadística descriptiva y preguntas de exploración — 5 puntos

Para la población analítica de 2025, calcule cantidad de observaciones, media, mediana, desviación estándar, mínimo, máximo, percentil 25, percentil 75 y percentil 95 de:

- Salario mensual.

- Edad.

- Antigüedad.

- Horas habituales de trabajo.

## Responda con evidencia gráfica:

- ¿Cómo se distribuyen los registros entre categorías ocupacionales, niveles educativos y dominios?

- ¿El salario presenta una distribución simétrica o asimétrica?

- ¿Qué diferencia existe entre su media y su mediana?

- ¿Cómo varía el salario mediano entre niveles educativos y categorías ocupacionales?

- ¿Cómo cambian el tamaño de la muestra analítica y el salario mediano entre trimestres?

Puede usar una escala logarítmica para visualizar la distribución, pero identifique claramente esa escala y conserve el objetivo original en quetzales para el modelado.


## 3. Relaciones entre variables numéricas — 5 puntos

Utilice VectorAssembler y Correlation.corr() de pyspark.ml.stat para obtener la correlación de Persona entre salario, edad, antigüedad y horas habituales. Presente una matriz de correlaciones con sus etiquetas y un mapa de calor.

## Responda:

- ¿Qué variables presentan mayor asociación lineal con el salario?

- ¿Existe relación entre edad y antigüedad?

Las correlaciones se calcularán sobre todos los registros elegibles de 2025.

## 4. Segmentación de perfiles mediante KMeans — 10 puntos

Construya una segmentación (clústering) usando las variables que considere adecuadas, analice si vale la pena incluir salario. Estandarice las variables y determine un número adecuado de K clústers (2, 3, 4 y 5). Eliga el mejor K según el criterio que tomen como grupo y asigne a cada cluster una descripción basada en sus resultados.

## MODELADO SUPERVISADO PARA PREDICCIÓN (75 pts)

Para esta etapa, la tarea consiste en estimar salario_mensual a partir de las seis variables indicadas. Es una predicción del salario del período observado, no un pronóstico individual del salario futuro.

## 5. Pipeline de regresión lineal — 20 puntos

Construya un pipeline de regresión que incluya:

- a. StringIndexer y OneHotEncoder para las variables categóricas: nivel educativo, categoría ocupacional y dominio.

- b. VectorAssembler para combinar las variables numéricas —edad, antigüedad y horas semanales— con las variables categóricas codificadas.

- c. Estandarización de los predictores, mediante la opción interna de LinearRegression o mediante StandardScaler. No es necesario aplicar ambos mecanismos.

- d. Un modelo LinearRegression para estimar el salario mensual.

Presente MAE, RMSE y R² e interprete los resultados frente al modelo de referencia.


Pruebe al menos dos configuraciones de regularización, documente los valores utilizados y seleccione la configuración con menor RMSE de validación. Ajuste todos los componentes del pipeline exclusivamente con los datos de entrenamiento. Guarde su mejor modelo, Reserve el primer trimestre de 2026 para la evaluación final de la actividad 7.

## 6. Pipeline de Random Forest — 20 puntos

Construya un pipeline de regresión que incluya:

- a. StringIndexer y OneHotEncoder para las variables categóricas: nivel educativo, categoría ocupacional y dominio.

- b. VectorAssembler para combinar las variables numéricas —edad, antigüedad y horas semanales— con las variables categóricas codificadas.

- c. Un modelo RandomForestRegressor para estimar el salario mensual. Este modelo no requiere estandarización de los predictores.

Utilice los mismos conjuntos de entrenamiento y validación de la actividad anterior. Presente MAE, RMSE y R² y compare los resultados con la regresión lineal y el modelo de referencia.

Pruebe al menos dos configuraciones, variando la cantidad de árboles y/o su profundidad máxima. Documente los valores utilizados y seleccione la configuración con menor RMSE de validación. Utilice una semilla fija y ajuste todos los componentes del pipeline exclusivamente con los datos de entrenamiento.

Explique cuál de los dos algoritmos obtuvo mejores resultados en validación y qué diferencias podrían explicar su desempeño. Guarde su mejor modelo, reserve el primer trimestre de 2026 para la evaluación final de la actividad 7.

## 7. Entrenamiento final y evaluación en 2026 — 20 puntos

Una vez seleccionada una configuración por algoritmo:

- 1. Genere predicciones sobre el primer trimestre de 2026, aplicando las reglas de preparación establecidas.

- 2. Evalúe los dos modelos sobre exactamente los mismos registros elegibles de prueba.

- 3. Compare los resultados con las mismas métricas establecidas.

## 8. Visualización y análisis de errores — 15 puntos


Universidad del Valle de Guatemala

Facultad de Ingeniería

Departamento de Ciencias de la Computación

CC3066 – Data Science

Para cada modelo entrenado, construya:

- 1. Un gráfico de salario real frente a salario predicho, con la línea de referencia (y=x).

- 2. Un gráfico de residuos frente al salario predicho, con una línea horizontal en cero.

- 3. Una tabla de MAE y error medio por nivel educativo y por dominio, incluyendo el número de observaciones de cada grupo.

Defina el residuo de forma consistente, un residuo positivo indica subestimación; uno negativo, sobreestimación. Use la misma muestra de hasta 5,000 registros para comparar visualmente ambos modelos. Calcule las métricas por grupo con todos los registros de prueba.

Será importante que analice el percentil de salarios y la distribución que encuentra en este sentido. Explique si los errores son y si existe una tendencia a subestimar o sobreestimar salarios altos. Incluya una discusión final englobando todos los hallazgos encontrados.

## EVALUACIÓN

- \- (25 puntos) Haber completado con exactitud la primera sección.

- \- (75 puntos) Haber respondido de manera correcta evidenciando el paso a paso de su solución para toda la segunda sección.

NOTA: para poder obtener derecho a nota, cada estudiante del grupo deberá haber entregado el ejercicio del día Lunes 21 de Septiembre.

## MATERIAL A ENTREGAR

- \- Script de Python (.ipynb) que utilizó con la discusión generada utilizando markdown.

- \- Link del repositorio usado para versionar el código.

NOTA: se deberá de evidenciar el trabajo colaborativo de todos los integrantes para obtar por calificación.

## FECHAS DE ENTREGA

- \- AVANCE: Jueves 24 de septiembre de 2026 a las 17:20hrs: Actividades correspondientes al análisis exploratorio y segmentación.

- \- ENTREGA FINAL COMPLETA: Domingo 27 de septiembre de 2026 a las 23:59.

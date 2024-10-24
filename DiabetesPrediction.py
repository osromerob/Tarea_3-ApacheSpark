import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, count, format_number
from pyspark.ml.feature import Imputer

# Crear una sesión de Spark
spark = SparkSession.builder.appName("DiabetesPrediction").getOrCreate()

# Ajustar la configuración para mostrar más campos
spark.conf.set("spark.sql.debug.maxToStringFields", 100)

# Cargar el conjunto de datos desde un archivo CSV
file_path = 'hdfs://localhost:9000/Tarea3/diabetes_dataset.csv'  # Cambia esto con la ruta real de tu archivo
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Obtener la ruta de la carpeta donde se ejecuta el script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Definir la ruta del archivo dinámicamente
output_path = os.path.join(current_dir, 'resultados.txt')

with open(output_path, 'w') as f:
    # Mostrar las primeras filas del DataFrame
    f.write("Primeras filas del DataFrame:\n")
    df.show(5)
    df.limit(5).toPandas().to_string(f)

    # Reemplazar los valores de "0" en las columnas específicas por "null"
    columns_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col_name in columns_to_fix:
        df = df.withColumn(col_name, when(col(col_name) == 0, None).otherwise(col(col_name)))

    # 1. Filtrar registros donde Glucose es mayor a 150
    high_glucose = df.filter(col("Glucose") > 150)
    f.write("\nRegistros con Glucose mayor a 150:\n")
    high_glucose.show(10)
    high_glucose.limit(10).toPandas().to_string(f)

    # 2. Filtrar registros donde Insulin es NULL
    null_insulin = df.filter(col("Insulin").isNull())
    f.write("\nRegistros con Insulina nula:\n")
    null_insulin.show(10)
    null_insulin.limit(10).toPandas().to_string(f)

    # 3. Filtrar registros donde BMI es menor a 18.5
    underweight_patients = df.filter(col("BMI") < 18.5)
    f.write("\nPacientes con BMI menor a 18.5:\n")
    underweight_patients.show(10)
    underweight_patients.limit(10).toPandas().to_string(f)

    # 4. Filtrar registros con BloodPressure mayor a 80 y ordenar por Glucose en orden descendente
    high_blood_pressure = df.filter(col("BloodPressure") > 80).sort(col("Glucose").desc())
    f.write("\nPacientes con presión arterial alta, ordenados por Glucosa:\n")
    high_blood_pressure.show(10)
    high_blood_pressure.limit(10).toPandas().to_string(f)

    # 5. Filtrar registros donde el resultado de diabetes (Outcome) es 1 y ordenar por BMI en orden ascendente
    diabetes_positive = df.filter(col("Outcome") == 1).sort(col("BMI").asc())
    f.write("\nPacientes con resultado positivo de diabetes, ordenados por BMI:\n")
    diabetes_positive.show(10)
    diabetes_positive.limit(10).toPandas().to_string(f)

    # 6. Filtrar registros donde Glucose es mayor a 200 y BMI es mayor a 30, y ordenar por Insulin
    high_risk_patients = df.filter((col("Glucose") > 200) & (col("BMI") > 30)).sort(col("Insulin").desc())
    f.write("\nPacientes de alto riesgo (Glucosa > 200 y BMI > 30), ordenados por Insulina:\n")
    high_risk_patients.show(10)
    high_risk_patients.limit(10).toPandas().to_string(f)

    # Resumen estadístico
    f.write("\nResumen estadístico del DataFrame:\n")
    summary = df.describe()
    summary.show()
    summary_pd = summary.toPandas()
    f.write(summary_pd.to_string())

    # Contar valores nulos por columna
    f.write("\nConteo de valores nulos por columna:\n")
    null_count = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
    null_count.show()
    null_count_pd = null_count.toPandas()
    f.write(null_count_pd.to_string())

    # Crear el imputador
    imputer = Imputer(inputCols=columns_to_fix, outputCols=[col+"_imputed" for col in columns_to_fix])

    # Ajustar el imputador y transformar los datos
    df_imputed = imputer.fit(df).transform(df)

    # Mostrar las primeras filas después de la imputación
    f.write("\nPrimeras filas después de la imputación:\n")
    df_imputed.show(5)
    df_imputed.limit(5).toPandas().to_string(f)

# Cerrar la sesión de Spark
spark.stop()

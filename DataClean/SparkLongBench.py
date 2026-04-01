import os
import sys
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import *


hadoop_bin = '/home/wangchanghui/hadoop/hadoop-3.4.0/bin'
os.environ['PATH'] = f"{hadoop_bin}:" + os.environ['PATH']
os.environ['SPARK_DIST_CLASSPATH'] = os.popen(f"{hadoop_bin}/hadoop classpath").read().strip()
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'
spark_home = '/home/wangchanghui/spark/spark-3.3.0-bin-hadoop3'
os.environ['SPARK_HOME'] = spark_home
os.environ['HADOOP_CONF_DIR'] = '/home/wangchanghui/hadoop/hadoop-3.4.0/etc/hadoop'
sys.path.insert(0, os.path.join(spark_home, 'python'))
os.environ['PYSPARK_PYTHON'] = '/home/wangchanghui/yes/envs/OCR/bin/python'

spark = (SparkSession.builder.master("yarn")
        .appName("LongBench_Clean")
        .config("spark.executor.memory", "3g")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.parquet.compression.codec","zstd")
        .config("spark.executor.cores", "2")
        .config("spark.default.parallelism", "200")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.yarn.jars", "hdfs://192.168.1.5:9000/spark-jars/spark-libs.jar")
        .config("spark.yarn.archive", "hdfs://192.168.1.5:9000/spark-jars/spark-libs.jar")
        .config("spark.hadoop.yarn.resourcemanager.address", "192.168.1.5:8032")
        .config("spark.hadoop.yarn.resourcemanager.hostname", "192.168.1.5")
        .config("spark.hadoop.fs.defaultFS", "hdfs://192.168.1.5:9000")
        .config("spark.yarn.stagingDir", "hdfs://192.168.1.5:9000/user/wangchanghui/.sparkStaging")
        .getOrCreate())

sc = spark.sparkContext
sc.setLogLevel("INFO")

chunk_schema = ArrayType(StringType())

data_input_path = "file:///home/wangchanghui/pycharm_projects/Ai_OCR/data/*.jsonl"
df_raw = spark.read.json(data_input_path)

@f.pandas_udf(chunk_schema)
def clean(inputs: pd.Series, contexts: pd.Series) -> pd.Series:
    raw_full = inputs.fillna("") + "\n\n" + contexts.fillna("")
    regex_cleaned = raw_full.str.replace(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', regex=True)
    clean_text = regex_cleaned.str.normalize('NFKC')
    chunks_series = clean_text.str.findall(r'[\s\S]{1,128}')
    return chunks_series.apply(lambda x: x if isinstance(x, list) else [])


df_repartitioned = df_raw.repartition(100)
df_step1 = df_repartitioned.withColumn("chunks_array", clean(f.col("input"), f.col("context")))
df_step2 = df_step1.withColumn("total_char_len", f.length(f.array_join(f.col("chunks_array"), "")))
df_step3 = df_step2.select(
    f.col("_id").alias("doc_id"),
    f.col("dataset").alias("task_type"),
    f.col("total_char_len"),
    f.posexplode("chunks_array").alias("pos", "chunk_str")
)
df_final = df_step3.withColumn(
    "pos_ratio",
    (f.col("pos") * 128) / f.col("total_char_len")
).select("doc_id", "task_type", "pos_ratio", "chunk_str")

output_path = 'hdfs://192.168.1.5:9000/user/data'
df_final.write.mode("overwrite").parquet(output_path)
print(f"结果已存入: {output_path}")

spark.stop()

# load a CSV file into apache spark
# run with spark-submit
# $ spark-submit spark-load-csv.py 
from pyspark.sql import SQLContext, SparkSession

DATA_FILE = "data/listings.csv"

spark = SparkSession.builder.appName("experiment").getOrCreate()

sqlContext = SQLContext(spark)

df = sqlContext.read.load(
        DATA_FILE, 
    format='com.databricks.spark.csv', 
    header='true', 
    inferSchema='true').cache()

print(df.columns);

spark.stop()

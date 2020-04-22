from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,when,max

if __name__ == '__main__':

    conf = SparkConf().setAppName("spark_app") \
        .setMaster("local[2]")

    ss = SparkSession.builder \
        .config(conf=conf).getOrCreate()

    df = ss.createDataFrame([
        ("a", 1, "m1"), ("a", 1, "m2"), ("a", 2, "m3"),
        ("a", 3, "m4"), ("b", 4, "m1"), ("b", 1, "m2"),
        ("b", 2, "m3"), ("c", 3, "m1"), ("c", 4, "m3"),
        ("c", 5, "m4"), ("d", 6, "m1"), ("d", 1, "m2"),
        ("d", 2, "m3"), ("d", 3, "m4"), ("d", 4, "m5"),
        ("e", 4, "m1"), ("e", 5, "m2"), ("e", 1, "m3"),
        ("e", 1, "m4"), ("e", 1, "m5")],
        ("a", "cnt", "major"))

    df.show()

    majors = sorted(df.select("major").distinct().rdd.map(lambda r:r[0]).collect())
    print(majors)

    cols = [when(col("major") == m, col("cnt")).otherwise(None).alias(m)
            for m in majors]
    maxs = [max(col(m)).alias(m) for m in majors]

    reshaped1 = (df
                 .select(col("a"), *cols)
                 .groupBy("a")
                 .agg(*maxs)
                 .na.fill(0))

    reshaped1.show()



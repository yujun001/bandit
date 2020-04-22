"""
基于tag 协同召回：
1.过去1天新用户喜欢相同tag 的所有用户;
2.这批用户完播或点赞top_n的视频召回,
"""
from pyspark.sql import SparkSession
import pyspark.sql.functions  as f
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, StructField, StructType, IntegerType, LongType, FloatType

# @F.pandas_udf(schema, F.PandasUDFType.GROUPED_MAP)
# def to_json(pdf):
#     return pdf.assign(serial=json.dumps({pdf.code: pdf.level}))

if __name__ == '__main__':

    print("test for line")

    spark = SparkSession\
        .builder\
        .appName('pyspark-by-examples')\
        .getOrCreate()

    # arrayData = [('James', ['Java', 'Scala'], {'hair': 'black', 'eye': 'brown'}),
    #     #              ('Michael', ['Spark', 'Java', None], {'hair': 'brown', 'eye': None}),
    #     #              ('Robert', ['CSharp', ''], {'hair': 'red', 'eye': ''}),
    #     #              ('Washington', None, None),
    #     #              ('Jefferson', ['1', '2'], {})]
    #     #
    #     # df = spark.createDataFrame(data=arrayData,
    #     #                            schema=['name', 'knownLanguages', 'properties'])

    l = [('a', 'foo', 1), ('b', 'bar', 1), ('a', 'biz', 6), ('c', 'bar', 3), ('c', 'biz', 2)]
    df = spark.createDataFrame(l, ('uid', 'code', 'level'))

    res = df.groupby('uid').agg(
        f.to_json(
            f.collect_list(
                f.create_map('code', 'level')
            )
        ).alias('json'))

    res.show()

    schema = StructType([StructField("k", StringType(), True),
                         StructField("v", IntegerType(), False)]
                        )
    rng = spark.createDataFrame([], schema=schema)
    rng.show()







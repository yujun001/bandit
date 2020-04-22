"""
分区域统计
各国家/指定推荐内容

"""
from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
import os
from datetime import datetime, timedelta
from pyspark.sql.types import StringType, StructField, StructType, IntegerType, LongType
from pyspark.sql.functions import col
import pyspark.sql.functions as F

ACTION_SCHEMA = StructType([
    StructField("gaid", StringType(), True),
    StructField("video_id", StringType(), True),
    StructField("type", IntegerType(), True),
    StructField("client_time", StringType(), True),
    StructField("counter", LongType(), True),
    StructField("country", StringType(), True)
])

ITEM_INFO = StructType([
    StructField("item_id", StringType(), True),
    StructField("duration", LongType(), True),
    StructField("tag", StringType(), True),
    StructField("country_code", StringType(), True),
    StructField("recommend", IntegerType(), True)
])

region_country = {
    "NG_RG": ['NG'],
    "GH_RG": ['GH'],
    "KE_RG": ['KE'],
    "TZ_RG": ['TZ'],
    "EAST_AF_RG": ['SS','ER','UG','RW','SC','ET'],
    "WEST_AF_EN_RG": ['GM','SL','LR','GW','CV','ST'],
    "SOUTH_AF_RG": ['ZM','BW','NA','ZA','SZ','LS','MU','MW','ZW','MZ','AO'],

    "CI_RG": ['CI'],
    "SN_RG": ['SN'],
    "FRA_RG": ['MG','KM','BI','DJ','TG','BJ','NE','ML','BF','GN','GA','CG','CD','TD','CF','CM','GQ'],

    "WHITE_ALB_RG": ['EH','EG','LY','TN','DZ','MA'],
    "BLACK_ALB_RG": ['SD','MR','SO'],

    "OTHER_RG": ['CA','CN','DE','FR','GB','HK','HU','IE','IN','IT','JP','PH','RU','SA','SE','TR','US']
}

rec_area = {
    "NG_REC_AREA": ['NG_RG','WEST_AF_EN_RG'],
    "GH_REC_AREA":['GH_RG','NG_RG','WEST_AF_EN_RG'],
    "KE_REC_AREA":['KE_RG','TZ_RG','EAST_AF_RG'],
    "TZ_REC_AREA":['TZ_RG','KE_RG','EAST_AF_RG'],

    "EAST_AF_REC_AREA":['EAST_AF_RG', 'KE_RG', 'TZ_RG'],
    "WEST_AF_EN_REC_AREA":['WEST_AF_EN_RG', 'NG_RG', 'GH_RG'],
    "SOUTH_AF_REC_AREA":['SOUTH_AF_RG'],

    "CI_REC_AREA": ['CI_RG','SN_RG','FRA_RG'],
    "SN_REC_AREA": ['SN_RG','CI_RG','FRA_RG'],
    "FRA_REC_AREA": ['FRA_RG','CI_RG','SN_RG'],

    "WHITE_ALB_REC_AREA": ['WHITE_ALB_RG'],
    "BLACK_ALB_REC_AREA": ['BLACK_ALB_RG'],

    "OTHER_REC_AREA": ['NG_RG', 'KE_RG', 'EAST_AF_RG', 'WEST_AF_EN_RG']
}

def get_data_paths_dt(base: str, start_tm: datetime, end_tm: datetime):
    """
    :param base:  action_log 根目录
    :param start_tm: 计算开始日期  2019-10-18 20:41:52.863955  sample
    :param end_tm:   计算结束日期  2019-10-22 20:41:52.863955
    :return:  action_log的文件夹列表
    """
    paths = []
    tmp_tm = start_tm
    while tmp_tm <= end_tm:
        tm_str = tmp_tm.strftime("dt=%Y%m%d")
        dir_str = os.path.join(base, tm_str)
        paths.append(dir_str)
        tmp_tm = tmp_tm + timedelta(days=1)
    return paths

def get_user_actoin(ss: SparkSession, tm_now, last_days, action_log_path) -> DataFrame:
    """
    :param tm_now: 当前日期
    :param last_days: 计算过去多少天日志 log
    :param action_log_path: 行为日志路径
    :return: df,  gaid, video_id, country, pg
    """
    start_tm = tm_now - timedelta(days=last_days)
    end_tm = tm_now + timedelta(days=-1)

    paths = get_data_paths_dt(action_log_path, start_tm, end_tm)
    df = ss.read.load(paths, schema=ACTION_SCHEMA)

    res = df.filter((col('gaid').isNotNull())
                    & (col('type').isin({2}))
                    & (col('gaid') != '00000000-0000-0000-0000-000000000000'))\
        .drop_duplicates()\
        .orderBy(['gaid', 'client_time'])\
        .drop('type')\
        .dropna()
    return res

def get_item_info(ss: SparkSession, data_path) -> DataFrame:
    paths = data_path
    df = ss.read.load(paths, schema=ITEM_INFO)

    res = df.filter(col('recommend') > 0)\
        .drop('recommend')\
        .withColumnRenamed('item_id', 'video_id')
    return res

def Complete_Compute(merge_action, region):
    """
    :param merge_action: df
    :param region: 各区域的tag
    :return:  计算各推荐区域的VV完播情况, 并保存到csv 文件
    """
    dir = "/Users/jun_yu/Downloads/action_log/"
    res_path = os.path.join(dir, region)

    action_RG = merge_action.filter(
        (col('country').isin(region_country.get(region))) & (col('country_code').isin(rec_area_user.get(region))))
    action_RG_Total = action_RG.groupBy(col('tag')).agg({'video_id': 'count'}) \
        .withColumnRenamed('count(video_id)', 'count_total')

    action_RG_Total_Complete = action_RG.filter(col('counter') >= 100).groupBy(col('tag')).agg({'video_id': 'count'}) \
        .withColumnRenamed('count(video_id)', 'count_total_complete')

    action_RG_Percent = action_RG_Total.join(action_RG_Total_Complete, on='tag', how='left').na.fill(0) \
        .withColumn('complete_percent', F.round(col('count_total_complete') * 1.0 / col('count_total'), 3))

    action_RG_Percent.sort(col('complete_percent'), ascending=False) \
        .repartition(1).write.format('com.databricks.spark.csv') \
        .save(res_path, header='true')
    print("compute have done %s" % region)
    return ''

if __name__ == '__main__':

    conf = SparkConf().setAppName("spark_app")\
        .setMaster("local[2]")

    ss = SparkSession.builder.config(conf=conf)\
        .getOrCreate()

    # 循环每个区域用户
    rec_area_user = {}
    for k, v in rec_area.items():
        tmp_key = k.split("_REC_AREA")[0] + "_RG"
        tmp_v = v
        list_res = []
        for v in tmp_v:
            res_v = region_country.get(v)
            list_res.extend(res_v)
        # 每个区域用户，可推荐内容的区域汇总
        rec_area_user[tmp_key] = list_res
    print(rec_area_user)
    # print(rec_area_user.get('NG'))

    last_days = 30
    tm_now = datetime.strptime('20191125', "%Y%m%d")
    # actionInput_dir = "/home/yujun/Downloads/tag_cluster"
    # itemInfo_dir = "/home/yujun/Downloads/item_info/dt=20191115"

    actionInput_dir = "/Users/jun_yu/Downloads/action_log"
    itemInfo_dir = "/Users/jun_yu/Downloads/item_info/dt=20191125"

    # last_days天的行为数据
    action_df = get_user_actoin(ss, tm_now, last_days, actionInput_dir)

    # 视频tag 信息处理
    item_df = get_item_info(ss, itemInfo_dir)
    item_df = item_df.na.drop(subset=['tag'])\
        .withColumn('tags', F.explode(F.split(col('tag'), ","))).drop('tag')\
        .withColumn('tag', F.explode(F.split(col('tags'), "/"))).filter(col('tag') != '1').drop('tags')\
        .withColumn('tags', F.when(col('tag') ==' Instrument', 'Instruments').otherwise(col('tag'))).drop('tag')\
        .withColumn('tag', F.when(col('tags') ==' bikes and boards', 'bikes and boards').otherwise(col('tags')))\
        .drop('tags')

    # 融合行为和视频的评分
    action_data = action_df.join(item_df, on='video_id', how='inner')
    merge_action = action_data.select(['gaid','video_id','counter','country','country_code','tag','duration','client_time'])

    for region in rec_area_user.keys():
        Complete_Compute(merge_action, region)


































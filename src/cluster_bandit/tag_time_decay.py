
"""
# 计算用户的兴趣
正向反馈,计算用户兴趣向量
counter: 可以选择 60, 90 等区间
加入时间衰减特征
"""

import datetime
from datetime import datetime, timedelta
from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
import os
from pyspark.sql.types import StringType, StructField, StructType, IntegerType, LongType
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, max
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np


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
    StructField("recommend", IntegerType(), True)
])

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
        .drop('type', 'country')\
        .dropna()

    return res

def get_item_info(ss: SparkSession, data_path) -> DataFrame:
    paths = data_path
    df = ss.read.load(paths, schema=ITEM_INFO)

    res = df.filter(col('recommend') > 0)\
        .drop('recommend')\
        .withColumnRenamed('item_id', 'video_id')
    return res


def get_launch_info(cluster_centers, launch_num):
    """
    获取试投视频的 Tag 及数量
    Args:
        cluster_centers: 聚类中心
        launch_num: 试探总视频个数
    Returns:
        launch_ratio: 各 Tag 试探视频的数量  {'Index_VideoTag': Amount, ...}
                    其中 Index: Tag 对应模型的索引
        major_tags: 主要的视频 Tag
    """
    cluster_cumstd = cluster_centers.std().sort_values(ascending=False)
    # print(cluster_cumstd)

    cluster_cumstd = (cluster_cumstd / cluster_centers.std().sum()).cumsum()
    major_tags = cluster_cumstd[cluster_cumstd <= 0.8].index[:5]

    major_tags = major_tags if len(major_tags) > 1 else cluster_cumstd.index[:2]  # 至少两个 tag

    launch_ratio = cluster_centers.std().sort_values(ascending=False)
    launch_ratio = (launch_ratio / cluster_centers.std().sum())[:len(major_tags)]
    launch_ratio = np.around(launch_ratio / launch_ratio.sum() * launch_num, 0)

    for _ in range(int(launch_ratio.sum() - launch_num)):
        launch_ratio[launch_ratio[launch_ratio != 0].index[-1]] -= 1
    inx = 0

    for _ in range(int(launch_num - launch_ratio.sum())):
        if (launch_ratio == 0).sum() != 0:
            launch_ratio[launch_ratio[launch_ratio == 0].index[0]] += 1
        else:
            launch_ratio[inx] += 1
            inx += 1

    if (launch_ratio != 0).sum() == 1:
        while launch_ratio[0] != 1 and (launch_ratio == 0).sum() != 0:
            launch_ratio[0] -= 1
            launch_ratio[launch_ratio[launch_ratio == 0].index[0]] += 1
            print(launch_ratio)

    major_tags = launch_ratio[launch_ratio != 0].index
    launch_ratio = launch_ratio[launch_ratio != 0]
    launch_ratio.index = [str(i + 1) + '_' + c for i, c in enumerate(launch_ratio.index)]
    launch_ratio = launch_ratio.to_dict()

    return launch_ratio, major_tags

def action_merge(actionInput_dir, itemInfo_dir, action_merge_path, last_days):

    # 近30天行为数据
    tm_now = datetime.strptime('20191115', "%Y%m%d")
    action_df = get_user_actoin(ss, tm_now, last_days, actionInput_dir)

    # 最新的视频信息 item_info
    item_df = get_item_info(ss, itemInfo_dir)
    
    # 视频tag 信息处理 
    item_df = item_df.na.drop(subset=['tag'])\
        .withColumn('tags', F.explode(F.split(col('tag'), ","))).drop('tag')\
        .withColumn('tag', F.explode(F.split(col('tags'), "/"))).filter(col('tag') != '1').drop('tags')\
        .withColumn('tags', F.when(col('tag') ==' Instrument', 'Instruments').otherwise(col('tag'))).drop('tag')\
        .withColumn('tag', F.when(col('tags') ==' bikes and boards', 'bikes and boards').otherwise(col('tags')))\
        .drop('tags')

    # 融合行为和视频的评分
    action_data = action_df.join(item_df, on='video_id', how='inner')

    action_log = action_data.withColumn("pg", F.when(col('duration') >= 10000, col('counter'))
                                        .otherwise(col('counter')*0.8))\
        .drop('counter', 'duration')\
        .filter(col('pg') >= 60)

    # 选取正样本, 且 进行过滤
    numViewsPerVideo = action_log.groupBy(col('video_id'))\
        .agg({'gaid': 'count'}).withColumnRenamed('count(gaid)', 'numViewsVideo')
    numViewsPerGaid = action_log.groupBy(col('gaid'))\
        .agg({'video_id': 'count'}).withColumnRenamed('count(video_id)', 'numViewsGaid')

    numViewsFilterGaid = numViewsPerGaid.filter((col('numViewsGaid') >= 100) & (col('numViewsGaid') <= 10000))
    numViewsFilterVideo = numViewsPerVideo.filter((col('numViewsVideo') >= 100) & (col('numViewsVideo') <= 100000))

    action = action_log.join(numViewsFilterGaid, on='gaid', how='inner')\
        .select('gaid', 'video_id', 'pg', 'tag')\
        .join(numViewsFilterVideo, on='video_id', how='inner')\
        .drop('numViewsVideo')

    action.repartition(64).write.parquet(action_merge_path)
    print("<<<<<<<<<<<<<user-item info contact done>>>>>>>>>>>>>>>")
    return action

def interst_feature(action_merge_path, feature_path):

    # 近30天的用户行为数据
    action_df = ss.read.load(action_merge_path)
    print("num of view: ", action_df.count())                                                # 20092699
    print("num of gaid: ", action_df.select(col('gaid')).drop_duplicates().count())          # 153664
    print("num of video_id: ", action_df.select(col('video_id')).drop_duplicates().count())  # 64177

    # 将行为转化为兴趣向量
    action_total = action_df.select('gaid','video_id','tag')\
        .groupBy(col('gaid'))\
        .agg({'tag':'count'})\
        .withColumnRenamed('count(tag)', 'total')

    action_tag_total = action_df.select('gaid','video_id','tag')\
        .groupBy('gaid','tag')\
        .agg({'video_id':'count'})\
        .withColumnRenamed('count(video_id)','tag_total')

    interest_df = action_total.join(action_tag_total, on='gaid', how='inner')

    interest_res = interest_df.withColumn('ration', F.round((col('tag_total')/col('total')), 3))

    tag_ration = interest_res.select('gaid', 'tag', 'ration')
    df = tag_ration.na.fill('other',subset=['tag'])

    majors = sorted(df.select("tag").distinct().rdd.map(lambda r:r[0]).collect())
    cols = [when(col("tag") == m, col("ration")).otherwise(F.lit('0')).alias(m) for m in majors]
    maxs = [max(col(m)).alias(m) for m in majors]

    reshaped1 = df.select(col("gaid"), *cols)\
        .groupBy("gaid")\
        .agg(*maxs)\
        .na.fill('0')

    reshaped1.repartition(1).write\
        .format('com.databricks.spark.csv')\
        .save(feature_path, header='true')
    print("<<<<<<<<<<<<<<<<<<<feature csv file done++++++++++++")
    return reshaped1

def cluster_for_user(k_cluster, df):
    # 对用户进行聚类
    kmeans = KMeans(init='k-means++',
                    n_clusters=k_cluster,
                    n_init=10,
                    random_state=9)
    pred_result = kmeans.fit_predict(df)

    # 每个类别的用户数
    cluster_num = pd.Series(pred_result).value_counts().sort_index().to_dict()
    print(cluster_num)
    print("the Distribution of Cluster Members[{}]:\n\t{}".format(k_cluster, cluster_num))
    return ''


def get_cluster_num(data, limit_cluster=10):
    print('the Clustering Results:')
    n_cluster = 0
    max_s_score = 0
    for cur_n_cluster in range(2, limit_cluster + 1):
        estimator = KMeans(init='k-means++',
                           n_clusters=cur_n_cluster,
                           n_init=10,
                           random_state=9)
        estimator.fit(data)
        s_score = metrics.silhouette_score(data,
                                           estimator.labels_,
                                           metric='euclidean',
                                           sample_size=5000,
                                           random_state=9)
        print('\tcluster:{:1d}  silhouette_score:{:.5f}'.format(cur_n_cluster, s_score))
        if s_score > max_s_score:
            n_cluster = cur_n_cluster
            max_s_score = s_score

    return n_cluster


if __name__ == '__main__':

    conf = SparkConf().setAppName("spark_app")\
        .setMaster("local[2]")

    ss = SparkSession.builder\
        .config(conf=conf).getOrCreate()

    # actionInput_dir = "/home/yujun/Downloads/tag_cluster"                        # 用户行为目录
    # itemInfo_dir = "/home/yujun/Downloads/item_info/dt=20191115"                 # item_info
    # last_days = 30                                                               # 过去30天行为
    # action_merge_path = "/home/yujun/Downloads/tag_cluster/action_time_decay/"   # 融合行为和item_info 的行为
    feature_path = "/home/yujun/Downloads/tag_cluster/action_time_decay_ration"  # 用户的兴趣向量

    # 获取用户过去30天行为,并融合 item信息
    # action_df = action_merge(actionInput_dir, itemInfo_dir, action_merge_path, last_days)
    # 将用户行为转换成 兴趣向量
    # feature = interst_feature(action_merge_path, feature_path)

    # interest_file = os.path.join(feature_path, "part-00000-09e393c3-dceb-40e7-a303-624e8b180141-c000.csv")
    interest_file = "/Users/jun_yu/Downloads/part-00000-09e393c3-dceb-40e7-a303-624e8b180141-c000.csv"   # desk_top

    df = pd.read_csv(interest_file)
    print(type(df))
    print(df.head(100))









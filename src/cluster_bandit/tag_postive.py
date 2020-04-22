"""
融合正负反馈下行为, 计算用户行为向量
counter 可以选择 <10, 大于 >90 作为正负反馈行为
"""
import datetime
from datetime import datetime, timedelta
from tqdm import tqdm
from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
import os
from pyspark.sql.types import StringType, StructField, StructType, IntegerType, LongType
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, max
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import tree
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
    # StructField("category", StringType(), True),
    StructField("tag", StringType(), True),
    StructField("recommend", IntegerType(), True)
])

def get_date_list(begin_date, total_days):
    begin_date = datetime.datetime.strptime(begin_date, "%Y%m%d")
    date_list = list()
    while total_days > 0:
        date_str = begin_date.strftime("%Y%m%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
        total_days -= 1
    return date_list


def get_log_data(begin_date, total_days):
    """
    获取从 begin_date 开始的 total_days 天的 log
     Args:
        begin_date: 开始时间
        total_days: 总天数
    Returns:
        data: dataframe, cols=['gaid', 'video_id', 'counter']
    """
    # path = r'E:\data\log_data'
    # path = "/Users/jun_yu/Downloads/action_log"
    date_list = get_date_list(begin_date, total_days)
    for cur_date in tqdm(date_list):
        print(cur_date)

    print(date_list)
    return ''

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
                    & ((col('counter') >= 90) | (col('counter') <= 10))
                    & (col('gaid') != '00000000-0000-0000-0000-000000000000'))\
        .withColumn("status", F.when((col('counter') >= 90), 1).otherwise(-1))\
        .drop_duplicates()\
        .orderBy(['gaid', 'client_time'])\
        .drop('type', 'client_time', 'counter')\
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

    """
    # 近30天行为数据, 进度超过90%的用户
    # actionInput_dir = "/Users/jun_yu/Downloads/action_log"
    actionInput_dir = "/home/yujun/Downloads/tag_cluster"
    tm_now = datetime.strptime('20191115', "%Y%m%d")
    last_days = 30
    actoin_df = get_user_actoin(ss, tm_now, last_days, actionInput_dir)

    # 最新的视频信息 item_info
    # itemInfo_dir = "/Users/jun_yu/Downloads/item_info/dt=20191115"
    itemInfo_dir = "/home/yujun/Downloads/item_info/dt=20191115"
    item_df = get_item_info(ss, itemInfo_dir)

    item_df = item_df.na.drop(subset=['tag'])\
        .withColumn('tags', F.explode(F.split(col('tag'), ","))).drop('tag')\
        .withColumn('tag', F.explode(F.split(col('tags'), "/"))).filter(col('tag') != '1').drop('tags')\
        .withColumn('tags', F.when(col('tag') ==' Instrument', 'Instruments').otherwise(col('tag'))).drop('tag')\
        .withColumn('tag', F.when(col('tags') ==' bikes and boards', 'bikes and boards').otherwise(col('tags')))\
        .drop('tags')

    # majors = sorted(item_df.select("tag").distinct().rdd.map(lambda r: r[0]).collect())
    # for i in majors:
    #     print(i)

    # 融合行为和视频的评分
    action_data = actoin_df.join(item_df, on='video_id', how='inner')
    action_data.show()

    # 选取样本(正负反馈)
    action_log = action_data

    numViewsPerVideo = action_log.groupBy(col('video_id'))\
        .agg({'gaid': 'count'}).withColumnRenamed('count(gaid)', 'numViewsVideo')

    numViewsPerGaid = action_log.groupBy(col('gaid'))\
        .agg({'video_id': 'count'}).withColumnRenamed('count(video_id)', 'numViewsGaid')

    numViewsPerVideo.repartition(1).write.format('com.databricks.spark.csv')\
        .save("/home/yujun/Downloads/tag_cluster/ViewsPerVideo", header='true')

    numViewsPerGaid.repartition(1).write.format('com.databricks.spark.csv')\
        .save("/home/yujun/Downloads/tag_cluster/ViewsPerGaid", header='true')

    numViewsFilterGaid = numViewsPerGaid.filter((col('numViewsGaid') >= 50) & (col('numViewsGaid') <= 10000))

    numViewsFilterVideo = numViewsPerVideo.filter((col('numViewsVideo') >= 50) & (col('numViewsVideo') <= 100000))

    # gaid|video_id|country|pg|category|tag
    action = action_log.join(numViewsFilterGaid, on='gaid', how='inner')\
        .select('gaid', 'video_id', 'status', 'tag')\
        .join(numViewsFilterVideo, on='video_id', how='inner')\
        .drop('numViewsVideo')

    # res_path = "/Users/jun_yu/Downloads/action_log/action_res/"
    res_path = "/home/yujun/Downloads/tag_cluster/action_merge/"
    action.repartition(64).write.parquet(res_path)
    print("<<<<<<<<<<<<<concat res done>>>>>>>>>>>>>>>")
    """

    """
    ######################
    # 近30天的用户行为数据
    # # res_path = "/Users/jun_yu/Downloads/action_log/action_res/"
    res_path = "/home/yujun/Downloads/tag_cluster/action_merge/"
    action_df = ss.read.load(res_path)
    # action_df.show()
    # action_df = action
    print("num of view: ", action_df.count())                                                # 20092699
    print("num of gaid: ", action_df.select(col('gaid')).drop_duplicates().count())          # 153664
    print("num of video_id: ", action_df.select(col('video_id')).drop_duplicates().count())  # 64177

    #####################
    # 将行为转化为兴趣向量
    action_total = action_df.select('gaid', 'video_id', 'tag')\
        .groupBy(col('gaid'))\
        .agg({'tag': 'count'})\
        .withColumnRenamed('count(tag)', 'total')
    action_total.show()

    action_total_status_1 = action_df.select('gaid', 'video_id', 'tag')\
        .filter(col('status') == 1)\
        .groupBy('gaid', 'tag')\
        .agg({'video_id': 'count'})\
        .withColumnRenamed('count(video_id)', 'tag_total_1')

    action_total_status_0 = action_df.select('gaid', 'video_id', 'tag')\
        .filter(col('status') == -1)\
        .groupBy('gaid', 'tag')\
        .agg({'video_id': 'count'})\
        .withColumnRenamed('count(video_id)', 'tag_total_0')

    interest_df_0 = action_total.join(action_total_status_0, on='gaid', how='inner').drop('total')
    interest_df_1 = action_total.join(action_total_status_1, on='gaid', how='inner').drop('total')
    interest = interest_df_0.join(interest_df_1, on=['gaid', 'tag'], how='full')\
        .na.fill(0)\
        .join(action_total, on='gaid', how='inner')\
        .withColumn('ration', F.round((col('tag_total_1')-col('tag_total_0'))/col('total'), 4))

    # interest_df_0.filter(col('gaid') == '000ea59a-1700-4f49-977a-5b915ea1d132').show(1000)
    # interest_df_1.filter(col('gaid') == '000ea59a-1700-4f49-977a-5b915ea1d132').show(1000)
    # interest.filter(col('gaid') == '000ea59a-1700-4f49-977a-5b915ea1d132').show(1000)

    # interest.show(10)

    tag_ration = interest.select('gaid', 'tag', 'ration')
    # df = tag_ration.na.fill('other', subset=['tag'])
    df = tag_ration
    majors = sorted(df.select("tag").distinct().rdd.map(lambda r: r[0]).collect())
    cols = [F.when(col("tag") == m, col("ration")).otherwise(F.lit(0)).alias(m) for m in majors]
    maxs = [F.sum(col(m)).alias(m) for m in majors]

    reshaped1 = df.select(col("gaid"), *cols)\
        .groupBy("gaid")\
        .agg(*maxs)\
        .na.fill(0)

    reshaped1.filter(col('gaid') == '000ea59a-1700-4f49-977a-5b915ea1d132').show()
    reshaped1.show(10)

    reshaped1.repartition(1).write\
        .format('com.databricks.spark.csv')\
        .save("/home/yujun/Downloads/tag_cluster/action_merge_ration/", header='true')
    
    """
    print("++++++++csv file output done++++++++++++")

    interest_file ="/home/yujun/Downloads/tag_cluster/action_merge_ration/part-00000-4a83a190-9378-4015-b29c-b920e09b56a0-c000.csv"
    df = pd.read_csv(interest_file)

    print(df.head())
    del df['gaid']

    # k_cluster = get_cluster_num(df)
    # print(k_cluster)

    k_cluster = 4

    ##############
    # 对用户进行聚类
    kmeans = KMeans(init='k-means++', n_clusters=k_cluster, n_init=10, random_state=9)

    kmeans.fit(df)
    
    pred_result = kmeans.fit_predict(df)
    cluster_num = pd.Series(pred_result).value_counts().sort_index().to_dict()

    print("the Distribution of Cluster Members[{}]:\n\t{}".format(k_cluster, cluster_num))
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = pd.DataFrame(cluster_centers, columns=df.columns)
    cluster_centers[cluster_centers < 0] = 0

    launch_num = 5
    # 寻找最具区分性的 Tag
    launch_ratio, major_tags = get_launch_info(cluster_centers, launch_num)
    print('the Main Tags of Video:', ', '.join(major_tags))
    print('the Number of Different Tags:\n\t', launch_ratio)

    user_tag = df
    # 构造分类特征
    user_tag = user_tag[major_tags]
    temp_data = user_tag.sum(axis=1)
    user_tag = user_tag[temp_data != 0]
    pred_result = pred_result[(temp_data != 0).values]
    temp_data = temp_data[temp_data != 0]
    user_tag = user_tag.apply(lambda x: x / temp_data)

    # 计算决策树分类准确率
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(user_tag, pred_result)
    acc_score = accuracy_score(pred_result, clf.predict(user_tag))
    print('Classification Accuracy: {:.2f}'.format(acc_score))



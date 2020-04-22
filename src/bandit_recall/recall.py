from datetime import datetime, timedelta
from pyspark.sql import SparkSession,DataFrame
from pyspark import SparkConf
from pyspark.sql.types import StringType, StructField, StructType, IntegerType, LongType
from pyspark.sql.functions import col, udf
import pyspark.sql.functions as F
import os


ACTION_SCHEMA = StructType([
    StructField("gaid", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("video_id", StringType(), True),
    StructField("type", IntegerType(), True),
    StructField("client_time", StringType(), True),
    StructField("counter", LongType(), True),
    StructField("country", StringType(), True)
])

ITEM_SCHEMA = StructType([
    StructField("item_id", StringType(), True),
    StructField("duration", LongType(), True),
    StructField("tag", StringType(), True),
    StructField("item_url", StringType(), True),
    StructField("country_code", StringType(), True),
    StructField("recommend", IntegerType(), True)
])

REC_HIS_SCHEMA = StructType([
    StructField("unique_id", StringType(), True),
    StructField("object_id", StringType(), True),
    StructField("pt_d", StringType(), True)  # rec_time
])


REGION_CONSIST_COUNTRY = {
    "NG_RG": ['NG'],
    "GH_RG": ['GH'],
    "KE_RG": ['KE'],
    "TZ_RG": ['TZ'],

    "EAST_AFRICA_RG": ['SS','ER','UG','RW','SC','ET'],
    "WEST_AFRICA_EN_RG": ['GM','SL','LR','GW','CV','ST'],
    "SOUTH_AFRICA_RG": ['ZM','BW','NA','ZA','SZ','LS','MU','MW','ZW','MZ','AO'],

    "CI_RG": ['CI'],
    "SN_RG": ['SN'],
    "FRANCE_RG": ['MG','KM','BI','DJ','TG','BJ','NE','ML','BF','GN','GA','CG','CD','TD','CF','CM','GQ'],

    "WHITE_ARAB_RG": ['EH','EG','LY','TN','DZ','MA'],
    "BLACK_ARAB_RG": ['SD','MR','SO'],

    "DEFAULT_RG": ['CA','CN','DE','FR','GB','HK','HU','IE','IN','IT','JP','PH','RU','SA','SE','TR','US']
}

REGION_REC_ITEM_REGION = {
    "NG_RG_REC": ['NG_RG','WEST_AFRICA_EN_RG'],
    "GH_RG_REC": ['GH_RG','NG_RG','WEST_AFRICA_EN_RG'],
    "KE_RG_REC": ['KE_RG','TZ_RG','EAST_AFRICA_RG'],
    "TZ_RG_REC": ['TZ_RG','KE_RG','EAST_AFRICA_RG'],

    "EAST_AFRICA_RG_REC": ['EAST_AFRICA_RG', 'KE_RG', 'TZ_RG'],
    "WEST_AFRICA_EN_RG_REC": ['WEST_AFRICA_EN_RG', 'NG_RG', 'GH_RG'],
    "SOUTH_AFRICA_RG_REC": ['SOUTH_AFRICA_RG'],

    "CI_RG_REC": ['CI_RG','SN_RG','FRANCE_RG'],
    "SN_RG_REC": ['SN_RG','CI_RG','FRANCE_RG'],
    "FRANCE_RG_REC": ['FRANCE_RG','CI_RG','SN_RG'],

    "WHITE_ARAB_RG_REC": ['WHITE_ARAB_RG'],
    "BLACK_ARAB_RG_REC": ['BLACK_ARAB_RG'],

    "DEFAULT_RG_REC": ['NG_RG', 'KE_RG', 'EAST_AFRICA_RG', 'WEST_AFRICA_EN_RG']
}


def region_rec_item_country(REGION_REC_ITEM_REGION, REGION_CONSIST_COUNTRY):
    """
    :param REGION_REC_ITEM_REGION:  每个区域可推荐内容所属区域
    :param REGION_CONSIST_COUNTRY:  每个区域包含的国家
    :return:  每个区域可推荐内容所属的国家
    """
    user_rec_item_country = {}
    for user_region, video_region in REGION_REC_ITEM_REGION.items():
        tmp_user_region = user_region.split("_REC")[0]
        tmp_video_region = video_region
        list_country = []
        for region in tmp_video_region:
            region_country = REGION_CONSIST_COUNTRY.get(region)
            list_country.extend(region_country)
        user_rec_item_country[tmp_user_region] = list_country
    return user_rec_item_country


def get_data_paths_dt(base: str, start_tm: datetime, end_tm: datetime):
    """
    get the dt path dir
    :param base:  日志log的 根目录
    :param start_tm: 计算开始日期  2019-10-18 20:41:52.863955  sample
    :param end_tm:   计算结束日期  2019-10-22 20:41:52.863955
    :return:  日志log文件夹列表
    """
    paths = []
    tmp_tm = start_tm
    while tmp_tm <= end_tm:
        tm_str = tmp_tm.strftime("dt=%Y%m%d")
        dir_str = os.path.join(base, tm_str)
        paths.append(dir_str)
        tmp_tm = tmp_tm + timedelta(days=1)
    return paths

def get_user_actoin(ss: SparkSession, tm_now, action_last_days, action_log_path) -> DataFrame:
    """
    get action log
    :param tm_now: 当前日期
    :param last_days: 计算过去多少天log
    :param action_log_path: 行为日志路径
    :return: df; gaid,user_id, video_id, country, counter, client_time
    """
    start_tm = tm_now - timedelta(days=action_last_days)
    end_tm = tm_now + timedelta(days=-1)

    paths = get_data_paths_dt(action_log_path, start_tm, end_tm)
    print("the action log path:%s" %paths)
    df = ss.read.load(paths, schema=ACTION_SCHEMA)

    res = df.filter((col('gaid').isNotNull())
                    & (col('type').isin({2}))
                    & (col('gaid') != '00000000-0000-0000-0000-000000000000'))\
        .drop_duplicates()\
        .orderBy(['gaid', 'client_time'])\
        .drop('type')\
        .dropna()

    return res

def get_item_info(ss: SparkSession, tm_now, data_path):
    """
    get the latest video_info
    :param ss: Spark
    :param data_path: video_detail_dir
    :return: df; dataframe
    """
    tmp_tm = tm_now + timedelta(days=-1)
    tm_str = tmp_tm.strftime("dt=%Y%m%d")
    base_dir = data_path
    paths = os.path.join(base_dir, tm_str)

    df = ss.read.load(paths, schema=ITEM_SCHEMA)

    res = df.filter(col('recommend') > 0)\
        .drop('recommend')\
        .withColumnRenamed('item_id', 'video_id')\
        .na.drop(subset=['tag'])\
        .withColumn('tags', F.explode(F.split(col('tag'), ","))).drop('tag')\
        .withColumn('tag', F.explode(F.split(col('tags'), "/"))).filter(col('tag') != '1').drop('tags')\
        .withColumn('tags', F.when(col('tag') ==' Instrument', 'Instruments').otherwise(col('tag'))).drop('tag')\
        .withColumn('tag', F.when(col('tags') ==' bikes and boards', 'bikes and boards').otherwise(col('tags')))\
        .drop('tags')

    return res

def get_rec_history(ss: SparkSession, tm_now, rec_his_last_days, rec_history_dir):
    """
    get rec_history
    :param tm_now: 当前日期
    :param last_days: 计算过去多少天rec log
    :param rec_history_dir: 推荐日志路径 rec_log
    :return: df, 含有 ['unique_id', 'video_id']字段
    """
    start_tm = tm_now - timedelta(days=rec_his_last_days)
    end_tm = tm_now + timedelta(days=-1)

    old_col_name = ['unique_id', 'object_id', 'pt_d']
    new_col_name = ['unique_id', 'video_id', 'recStartTime']
    mapping = dict(zip(old_col_name, new_col_name))

    paths = get_data_paths_dt(rec_history_dir, start_tm, end_tm)
    rec_his_df = ss.read.load(paths) \
        .select([col(c).alias(mapping.get(c)) for c in old_col_name]) \
        .groupBy(col('unique_id')) \
        .agg(F.collect_list('video_id').alias('rec_history'))

    return rec_his_df

def group_func(country, country_code):
    """
    :param country:  user region
    :param country_code: video region
    :return:  user-video  region group
    """
    REC_RG = [RG for RG in REGION_CONSIST_COUNTRY.keys()]
    group = "None_Group"
    for i in REC_RG:
        if (country in REGION_CONSIST_COUNTRY.get(i)) and (country_code in USER_REGION_REC_ITEM_COUNTRY.get(i)):
            group = i
    return group

udf_group = udf(group_func, StringType())

def action_group_job(actionInput, iteminfoInput, rechisInput, tag_explor):
    """
    :param tag:  待召回的标签tag
    :param action_df: 过去3天的action历史
    :param rec_his_df: 过去30天的推荐历史
    :return: 各指定tag对应用户组group, 过去3天的top-30条video信息
    """
    action_last_days = 3
    rec_his_last_days = 3
    action_df = get_user_actoin(ss, tm_now, action_last_days, actionInput)
    item_df = get_item_info(ss, tm_now, iteminfoInput)
    rec_his_df = get_rec_history(ss, tm_now, rec_his_last_days, rechisInput)

    merge_action = action_df.join(item_df, on='video_id', how='inner')\
        .select(['gaid', 'user_id', 'video_id', 'counter', 'country', 'country_code', 'tag', 'item_url', 'duration', 'client_time'])\
        .filter(col('country').isNotNull() & col('country_code').isNotNull())

    merge_df = merge_action.withColumn('group_name',
                                       udf_group(col('country'), col('country_code')))\
        .filter(col('group_name') != 'None_Group')

    return merge_df


def tag_recall_job(df, tag):
    """
    获取相同tag的用户id,
    然后找到这批用户所有观看视频中完播率top30的;
    :param df: 各个group 下近3天的观看历史
    :return: 各个group下, 不同tag的视频召回集合
    """
    print("+++++++++")
    # tag_gaid = df.filter(col('tag') =='solo comedy').select('gaid').drop_duplicates()
    tag_gaid = df.filter(col('tag') == tag).select('gaid').drop_duplicates()
    tag_gaid.show()

    tag_gaid_action = df.join(tag_gaid, on='gaid', how='inner')

    tag_gaid_action_Total = tag_gaid_action\
        .groupBy(col('video_id'))\
        .agg({'gaid': 'count'}) \
        .withColumnRenamed('count(gaid)', 'count_total')

    tag_gaid_action_Total_Complete = tag_gaid_action.filter(col('counter') >= 100)\
        .groupBy(col('video_id'))\
        .agg({'gaid': 'count'}) \
        .withColumnRenamed('count(gaid)', 'count_total_complete')

    tag_gaid_action_Percent = tag_gaid_action_Total.join(tag_gaid_action_Total_Complete, on='video_id', how='left')\
        .na.fill(0) \
        .withColumn('complete_percent', F.round(col('count_total_complete') * 1.0 / col('count_total'), 3))\
        .filter(col('count_total') >= 300)\
        .drop('count_total_complete')\
        .drop_duplicates()

    tag_df = df.select('video_id','tag','duration','item_url').drop_duplicates()

    recall_video_df = tag_gaid_action_Percent.join(tag_df, on='video_id', how='inner')\
        .select('video_id', 'count_total', 'complete_percent', 'tag', 'duration', 'item_url')\
        .orderBy([col('complete_percent'), col('count_total')], ascending=False)

    recall_video_df.show(50)

    return recall_video_df

if __name__ == "__main__":

    # 推荐历史不用考虑了......
    # tm_now = datetime.now()
    tm_now = datetime.strptime('20191128', "%Y%m%d")
    conf = SparkConf().setAppName("spark_app").setMaster("local[2]")
    ss = SparkSession.builder.config(conf=conf).getOrCreate()

    actionInput =   "/Users/jun_yu/Downloads/bandit_action/"
    iteminfoInput = "/Users/jun_yu/Downloads/bandit_item_info/"
    rechisInput =   "/Users/jun_yu/Downloads/bandit_rec_his/"
    group_log = "/Users/jun_yu/Downloads/bandit_group_action/"
    group_path = os.path.join(group_log, tm_now.strftime("dt=%Y%m%d"))

    tag_explor = ['drama', 'afrobeat dance','Duet & React','solo comedy',
                  'lip sync','Clothing and accessories','others(life - style)',
                  'World food','Skills & Creative','others(fashion)','magic',
                  'other dance','Athletics and skills','football','baby','cooking',
                  'Afro food','Animal in nature','love','makeup','hairstyle',
                  'Classic dance','Group dance','fitness','design']

    USER_REGION_REC_ITEM_COUNTRY = region_rec_item_country(REGION_REC_ITEM_REGION, REGION_CONSIST_COUNTRY)

    # group_df = action_group_job(actionInput,
    #                             iteminfoInput,
    #                             rechisInput,
    #                             tag_explor,)
    # group_df.repartition(128).write.parquet(group_path)

    # 各分组group的 tag 协同召回 ,
    # 计算视频完播率时, 播放进度做抑制 小于10s的counter 衰减系数0.8
    group_df = ss.read.parquet(group_path)
    group_df_NG = group_df.filter(col('group_name') == 'NG_RG')

    # 每个区域分别做不同tag的召回, 将召回结果分别保存
    for tag in tag_explor:
        recall_path = os.path.join(group_log, "recall_video", tag)
        print(tag, recall_path)
        recall_video_df = tag_recall_job(group_df_NG, tag)

        recall_video_df.repartition(1)\
            .write.format('com.databricks.spark.csv')\
            .save(recall_path, header='true')

    ss.stop()

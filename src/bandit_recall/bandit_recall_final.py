from datetime import datetime, timedelta
from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf
from pyspark.sql.types import StringType, StructField, StructType, IntegerType, LongType, FloatType, Row
from pyspark.sql.functions import col, collect_list,struct
import pyspark.sql.functions as F
import os
import math
from pyspark.sql.window import *

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

# 各区域包含的国家
REGION_CONSIST_COUNTRY = {
    "NG_RG": ['NG'],
    "GH_RG": ['GH'],
    "KE_RG": ['KE'],
    "TZ_RG": ['TZ'],

    "EAST_AFRICA_RG": ['SS', 'ER', 'UG', 'RW', 'SC', 'ET'],
    "WEST_AFRICA_EN_RG": ['GM', 'SL', 'LR', 'GW', 'CV', 'ST'],
    "SOUTH_AFRICA_RG": ['ZM', 'BW', 'NA', 'ZA', 'SZ', 'LS', 'MU', 'MW', 'ZW', 'MZ', 'AO'],

    "CI_RG": ['CI'],
    "SN_RG": ['SN'],
    "FRANCE_RG": ['MG', 'KM', 'BI', 'DJ', 'TG', 'BJ', 'NE', 'ML', 'BF', 'GN', 'GA', 'CG', 'CD', 'TD', 'CF', 'CM', 'GQ'],

    "WHITE_ARAB_RG": ['EH', 'EG', 'LY', 'TN', 'DZ', 'MA'],
    "BLACK_ARAB_RG": ['SD', 'MR', 'SO'],

    "DEFAULT_RG": ['CA', 'CN', 'DE', 'FR', 'GB', 'HK', 'HU', 'IE', 'IN', 'IT', 'JP', 'PH', 'RU', 'SA', 'SE', 'TR', 'US']
}

# 各区域可推荐内容所属区域
REGION_REC_ITEM_REGION = {
    "NG_RG_REC": ['NG_RG', 'WEST_AFRICA_EN_RG'],
    "GH_RG_REC": ['GH_RG', 'NG_RG', 'WEST_AFRICA_EN_RG'],
    "KE_RG_REC": ['KE_RG', 'TZ_RG', 'EAST_AFRICA_RG'],
    "TZ_RG_REC": ['TZ_RG', 'KE_RG', 'EAST_AFRICA_RG'],

    "EAST_AFRICA_RG_REC": ['EAST_AFRICA_RG', 'KE_RG', 'TZ_RG'],
    "WEST_AFRICA_EN_RG_REC": ['WEST_AFRICA_EN_RG', 'NG_RG', 'GH_RG'],
    "SOUTH_AFRICA_RG_REC": ['SOUTH_AFRICA_RG'],

    "CI_RG_REC": ['CI_RG', 'SN_RG', 'FRANCE_RG'],
    "SN_RG_REC": ['SN_RG', 'CI_RG', 'FRANCE_RG'],
    "FRANCE_RG_REC": ['FRANCE_RG', 'CI_RG', 'SN_RG'],

    "WHITE_ARAB_RG_REC": ['WHITE_ARAB_RG'],
    "BLACK_ARAB_RG_REC": ['BLACK_ARAB_RG'],

    "DEFAULT_RG_REC": ['NG_RG', 'KE_RG', 'EAST_AFRICA_RG', 'WEST_AFRICA_EN_RG']
}

# 各区域tag候选集(12个区域,除黑阿拉伯地区 BLACK_ARAB)
RG_TAG = {
    "NG_RG_TAG": ['drama',
                  'afrobeat dance',
                  'Duet & React',
                  'solo comedy',
                  'lip sync',
                  'Clothing and accessories',
                  'others(life-style)',
                  'World food',
                  'Skills & Creative',
                  'others(fashion)',
                  'magic',
                  'other dance',
                  'skill and competition',
                  'football',
                  'baby',
                  'cooking',
                  'Afro food',
                  'Animal in nature',
                  'love',
                  'makeup',
                  'hairstyle',
                  'Classic dance',
                  'Group dance',
                  'fitness',
                  'design'],

    "GH_RG_TAG": ['drama',
                  'Duet & React',
                  'afrobeat dance',
                  'solo comedy',
                  'lip sync',
                  'Clothing and accessories',
                  'others(life-style)',
                  'World food',
                  'Skills & Creative',
                  'others(fashion)',
                  'magic',
                  'baby',
                  'skill and competition',
                  'other dance',
                  'fitness',
                  'Afro food',
                  'Animal in nature',
                  'love',
                  'hairstyle',
                  'makeup',
                  'football',
                  'Cooking',
                  'Classic dance',
                  'others(sport)',
                  'design'],

    "KE_RG_TAG": ['drama',
                  'solo comedy',
                  'others(life-style)',
                  'Duet & React',
                  'World food',
                  'Clothing and accessories',
                  'magic',
                  'afrobeat dance',
                  'lip sync',
                  'beauty',
                  'others(fashion)',
                  'Hip Hop dance',
                  'love',
                  'Skills & Creative',
                  'painting',
                  'fitness',
                  'city scenery',
                  'design',
                  'Work & Business',
                  'hairstyle',
                  'baby',
                  'handwork',
                  'Swimming',
                  'makeup',
                  'skill and competition'],

    "TZ_RG_TAG": ['drama',
                  'afrobeat dance',
                  'solo comedy',
                  'World food',
                  'magic',
                  'others(life-style)',
                  'design',
                  'Clothing and accessories',
                  'Duet & React',
                  'fitness',
                  'city scenery',
                  'pet',
                  'Scenery',
                  'painting',
                  'skill and competition',
                  'baby',
                  'Skills & Creative',
                  'lip sync',
                  'Hip Hop dance',
                  'beauty',
                  'love',
                  'hairstyle',
                  'handwork',
                  'makeup',
                  'Group dance'],

    "EAST_AFRICA_RG_TAG": ['drama',
                           'afrobeat dance',
                           'solo comedy',
                           'design',
                           'others(life-style)',
                           'World food',
                           'magic',
                           'Duet & React',
                           'Clothing and accessories',
                           'fitness',
                           'city scenery',
                           'pet',
                           'painting',
                           'skill and competition',
                           'Skills & Creative',
                           'baby',
                           'Cooking',
                           'lip sync',
                           'beauty',
                           'Hip Hop dance',
                           'love',
                           'Scenery',
                           'Swimming',
                           'handwork',
                           'hairstyle'],

    "WEST_AFRICA_EN_RG_TAG": ['drama',
                              'Duet & React',
                              'afrobeat dance',
                              'solo comedy',
                              'Clothing and accessories',
                              'World food',
                              'Skills & Creative',
                              'others(fashion)',
                              'magic',
                              'baby',
                              'lip sync',
                              'others(life-style)',
                              'Hip Hop dance',
                              'beauty',
                              'football',
                              'other dance',
                              'skill and competition',
                              'fitness',
                              'Afro food',
                              'Animal in nature',
                              'love',
                              'hairstyle',
                              'Cooking',
                              'design',
                              'Classic dance'],

    "SOUTH_AFRICA_RG_TAG": ['Duet & React',
                            'drama',
                            'afrobeat dance',
                            'Clothing and accessories',
                            'love',
                            'fitness',
                            'baby',
                            'lip sync',
                            'hand tutting',
                            'solo comedy',
                            'Hip Hop dance',
                            'beauty',
                            'skill and competition',
                            'others(fashion)',
                            'other dance',
                            'Group dance',
                            'sing',
                            'Classic dance',
                            'others(life-style）',
                            'Skills & Creative',
                            'pet '],

    "CI_RG_TAG": ['drama',
                  'Duet & React',
                  'afrobeat dance',
                  'solo comedy',
                  'makeup',
                  'Clothing and accessories',
                  'Skills & Creative',
                  'Hip Hop dance',
                  'design',
                  'lip sync',
                  'beauty',
                  'magic',
                  'other dance',
                  'others(life-style)',
                  'skill and competition',
                  'hairstyle',
                  'others(fashion)',
                  'fitness',
                  'baby',
                  'painting',
                  'Group dance',
                  'others(sport)',
                  'love',
                  'Instruments',
                  'hand tutting'],

    "SN_RG_TAG": ['drama',
                  'Duet & React',
                  'solo comedy',
                  'makeup',
                  'lip sync',
                  'afrobeat dance',
                  'Clothing and accessories',
                  'Skills & Creative',
                  'beauty',
                  'football',
                  'others(life-style)',
                  'magic',
                  'design',
                  'fitness',
                  'baby',
                  'skill and competition',
                  'hairstyle',
                  'Hip Hop dance',
                  'others(fashion)',
                  'other dance',
                  'sing',
                  'painting',
                  'Group dance',
                  'love',
                  'flaunt'],

    "FRANCE_RG_TAG": ['drama',
                      'Duet & React',
                      'afrobeat dance',
                      'makeup',
                      'beauty',
                      'Clothing and accessories',
                      'Skills & Creative',
                      'hairstyle',
                      'design',
                      'Hip Hop dance',
                      'magic',
                      'fitness',
                      'lip sync',
                      'solo comedy',
                      'football',
                      'others(life-style)',
                      'skill and competition',
                      'others(fashion)',
                      'baby',
                      'other dance',
                      'painting',
                      'Group dance',
                      'hand tutting',
                      'others(sport)',
                      'love'],

    "WHITE_ARAB_RG_TAG": ['drama',
                          'solo comedy',
                          'skill and competition',
                          'Duet & React',
                          'painting',
                          'lip sync',
                          'Group dance',
                          'love',
                          'Skills & Creative',
                          'Scenery',
                          'Clothing and accessories',
                          'others(life-style)',
                          'other dance',
                          'beauty',
                          'sing',
                          'others(fashion)',
                          'fitness',
                          'Hip Hop dance',
                          'magic',
                          'others(sport)',
                          'baby',
                          'football',
                          'afrobeat dance',
                          'hand tutting',
                          'design'],

    "DEFAULT_RG_TAG": ['drama',
                       'solo comedy',
                       'Duet & React',
                       'afrobeat dance',
                       'lip sync',
                       'Clothing and accessories',
                       'others(life-style)',
                       'others(fashion)',
                       'World food',
                       'Skills & Creative',
                       'magic',
                       'Hip Hop dance',
                       'beauty',
                       'love',
                       'skill and competition',
                       'baby',
                       'Cooking',
                       'other dance',
                       'football',
                       'Afro food',
                       'city scenery',
                       'makeup',
                       'hairstyle',
                       'fitness',
                       'design']
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
    get the action log
    :param tm_now: 当前日期
    :param last_days: 计算过去多少天log
    :param action_log_path: 行为日志路径
    :return: df; gaid,user_id, video_id, country, counter, client_time
    """
    start_tm = tm_now - timedelta(days=action_last_days)
    end_tm = tm_now + timedelta(days=-1)

    paths = get_data_paths_dt(action_log_path, start_tm, end_tm)
    print("the action log path:%s" % paths)
    df = ss.read.load(paths, schema=ACTION_SCHEMA)

    res = df.filter((col('gaid').isNotNull())
                    & (col('type').isin({2}))
                    & (col('gaid') != '00000000-0000-0000-0000-000000000000')) \
        .drop_duplicates() \
        .orderBy(['gaid', 'client_time']) \
        .drop('type') \
        .dropna()

    return res


def  get_item_info(ss: SparkSession, tm_now, data_path):
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

    res = df.filter(col('recommend') > 0) \
        .drop('recommend') \
        .withColumnRenamed('item_id', 'video_id') \
        .na.drop(subset=['tag']) \
        .withColumn('tags', F.explode(F.split(col('tag'), ","))).drop('tag') \
        .withColumn('tag', F.explode(F.split(col('tags'), "/"))).filter(col('tag') != '1').drop('tags') \
        .withColumn('tags', F.when(col('tag') == ' Instrument', 'Instruments').otherwise(col('tag'))).drop('tag') \
        .withColumn('tag', F.when(col('tags') == ' bikes and boards', 'bikes and boards').otherwise(col('tags'))) \
        .drop('tags')
    return res


@F.udf(returnType=FloatType())
def walson_score(num_vv, num_complete):
    """
    :param num_vv:  总vv数
    :param num_complete: 完播数
    :return: 对小样本做了抑制的walson_score
    """
    if (num_vv * num_complete == 0) or (num_vv < num_complete):
        score = 0
    else:
        z = 1.96  # 置信度参数
        n = num_vv
        p = 1.0 * num_complete / num_vv
        score = (p + z * z / (2.0 * n) - z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / (1.0 + z * z / n)
    return score


@F.udf(returnType=StringType())
def group_func(country, country_code):
    """
    assign  group name for each action log
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

def action_group_job(actionInput, iteminfoInput):
    """
    :param tag:  待召回的标签tag
    :param action_df: 过去3天的action历史
    :return: 根据user-vieo 地区信息，将action 用户分组
    """
    action_last_days = 3
    action_df = get_user_actoin(ss, tm_now, action_last_days, actionInput)
    item_df = get_item_info(ss, tm_now, iteminfoInput)

    merge_action = action_df.join(item_df, on='video_id', how='inner') \
        .select(['gaid', 'user_id', 'video_id', 'counter', 'country',
                 'country_code', 'tag', 'item_url', 'duration', 'client_time']) \
        .filter(col('country').isNotNull() & col('country_code').isNotNull())

    merge_df = merge_action.withColumn('group_name', group_func(col('country'), col('country_code'))) \
        .filter(col('group_name') != 'None_Group')
    return merge_df

def video_score_job(df, tag):
    """
    指定group, 指定tag下, 协同用户过去3天 top视频召回(walson分数)
    :param df: 指定group 下近3天的观看历史
    :return: 指定group, 指定tag的watson完播率 计算
    """
    # 取gaid, 看过相同tag
    tag_gaid = df.filter(col('tag') == tag).select('gaid').drop_duplicates()

    # 对应gaid所有的 action
    tag_gaid_action = df.join(tag_gaid, on='gaid', how='inner')

    # 上述action 对应视频video_id 被观看数
    tag_gaid_action_Total = tag_gaid_action \
        .groupBy(col('video_id')) \
        .agg({'gaid': 'count'}) \
        .withColumnRenamed('count(gaid)', 'count_total')

    # 上述action 对应视频video_id 被完播数
    tag_gaid_action_Total_Complete = tag_gaid_action.filter(col('counter') >= 100) \
        .groupBy(col('video_id')) \
        .agg({'gaid': 'count'}) \
        .withColumnRenamed('count(gaid)', 'count_total_complete')

    # 观看 join 完播，求top walson_score 分数
    tag_gaid_action_Percent = tag_gaid_action_Total.join(tag_gaid_action_Total_Complete, on='video_id', how='left') \
        .na.fill(0) \
        .withColumn('complete_percent', F.round(walson_score(col('count_total'), col('count_total_complete')), 3)) \
        .filter(col('count_total') >= 100) \
        .drop('count_total_complete') \
        .drop_duplicates()

    recall_video_df = tag_gaid_action_Percent.select('video_id', 'complete_percent')\
        .withColumn('recall_tag', F.lit(tag))

    recall_video_df = recall_video_df \
        .withColumn('row_number',
                    F.row_number().over(Window.partitionBy(col('recall_tag')).orderBy(F.desc('complete_percent')))) \
        .filter(col('row_number') < 50) \
        .drop('row_number')

    return recall_video_df

def lambda_video_score(r: Row):
    video_detail = r['actions']
    dict = {}
    for i in range(len(video_detail)):
        video = video_detail[i]['video_id']
        score = round(video_detail[i]['complete_percent'],4)
        dict[video] = score
    m = dict
    return m

def recall_job(group_df, result_path):
    """
    :param group_df: 分组group_name 后的action_log
    :return:
    """
    REC_RG = [RG for RG in REGION_CONSIST_COUNTRY.keys() if RG != 'BLACK_ARAB_RG']  # 区域group, 除黑阿拉伯
    recall_total = [{}]
    for region in REC_RG:
        region_tag = region + "_TAG"
        recall_list = [{}]
        i = 0
        # 遍历每个分组内tag, 按tag做协同召回
        for tag in RG_TAG.get(region_tag):
            i +=1
            print(i)
            df = group_df.filter(col('group_name') == region)
            recall_res = video_score_job(df, tag)
            res = recall_res\
                .groupBy('recall_tag').agg(collect_list(struct(*['video_id','complete_percent'])).alias("actions"))\
                .rdd\
                .map(lambda r: Row(region=region.split('_RG')[0], tag=tag, recall_list=lambda_video_score(r)))
            value = res.collect()
            if len(value) > 0:
                res_json = res.collect()[0].asDict()
                recall_list.append(res_json)
                print(len(recall_list))
        recall_total.extend(recall_list)
        print(len(recall_total))
    # 所有地区, tag的召回结果存储
    ss.sparkContext.parallelize(recall_total).repartition(64).saveAsTextFile(result_path)
    return ''


if __name__ == "__main__":

    # tm_now = datetime.now()
    tm_now = datetime.strptime('20191128', "%Y%m%d")
    conf = SparkConf().setAppName("spark_app").setMaster("local[2]")
    ss = SparkSession.builder.config(conf=conf).getOrCreate()

    # action base dir
    actionInput = "/Users/jun_yu/Downloads/bandit_action/"
    # latest item_info
    iteminfoInput = "/Users/jun_yu/Downloads/bandit_item_info/"
    # recall result base dir
    group_log = "/Users/jun_yu/Downloads/bandit_group/"
    result_path = os.path.join(group_log, 'final_recall', tm_now.strftime("dt=%Y%m%d"))

    # 用户可推荐区域计算
    USER_REGION_REC_ITEM_COUNTRY = region_rec_item_country(REGION_REC_ITEM_REGION, REGION_CONSIST_COUNTRY)

    # 分区域,tag的协同结果, 并存储
    group_df = action_group_job(actionInput, iteminfoInput)
    recall_job(group_df, result_path)

    ss.stop()

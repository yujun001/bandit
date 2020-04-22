# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:48:42 2019

@author: tn_jinfeng.zhang
"""

import os
import json
import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml

area_table = {
    'AFRICA_EN': ["NG", "GH", "GM", "SL", "LR", "KE", "TZ", "SS", "UG", "ZA", "BW", "NA", "SZ", "LS", "MW", "ZM", "ZW"],
    'AFRICA_FR': ["CI", "SN", "ML", "BF", "GQ", "TG", "BJ", "NE", "BI", "CF", "GQ", "GA", "CG", "CD", "MG"],
    'AFRICA_EF': ["RW", "SC", "CM", "MU"],
    'AFRICA_FA': ["RW", "SC", "CM", "MU"],
    'AFRICA_EA': ["ER"],
    'AFRICA_AB': ["MR", "SO"],
    'AFRICA_SM': ["GW", "CV", "ET", "ST", "AO", "MZ"],
    'NORTH_AFRICA_AB': ["EH", "EG", "SD", "LY", "TN", "DZ", "MA"]}

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
    path = r'E:\data\log_data'
    date_list = get_date_list(begin_date, total_days)
    data = list()
    for cur_date in tqdm(date_list):
        for cur_time in ['AM', 'PM']:
            cur_file = os.path.join(path, 'log_{}_{}.csv'.format(cur_date, cur_time))
            cur_data = pd.read_csv(cur_file, usecols=['gaid', 'video_id', 'type', 'create_time', 'counter', 'country'])
            cur_data = cur_data[cur_data['type'] == 2]
            cur_data.dropna(inplace=True)
            cur_data.drop('type', axis=1, inplace=True)
            data.append(cur_data)
    data = pd.concat(data)
    data['create_time'] = pd.to_datetime(data['create_time'])
    data.drop_duplicates(inplace=True)
    data = data[data['counter'] >= 5]
    data.loc[data['counter'] > 100, 'counter'] = 100
    data = data[data['gaid'] != '00000000-0000-0000-0000-000000000000']
    data.sort_values(['gaid', 'create_time'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.drop('create_time', axis=1, inplace=True)
    return data


def get_video_tag():
    """
    获取视频标签信息
    Args:
        --
    Returns:
        video_tag: dataframe, cols=['video_id', 'title', 'tags']
    """
    path = r'E:\data\log_data'
    file_path = os.path.join(path, 'video_tags_info.csv')
    video_tag = pd.read_csv(file_path, usecols=['video_id', 'title', 'tags'])
    video_tag.drop_duplicates(keep='last', inplace=True)
    video_tag.dropna(inplace=True)
    return video_tag


def get_cluster_num(data, limit_cluster=5):
    """
    用轮廓系数选择合适的簇数
    Args:
        data: 聚类数据
        limit_cluster: 最大聚类个数
    Returns:
        n_cluster: 最佳聚类个数
    """
    print('the Clustering Results:')
    n_cluster = 0
    max_s_score = 0
    for cur_n_cluster in range(2, limit_cluster + 1):
        estimator = KMeans(init='k-means++', n_clusters=cur_n_cluster, n_init=10, random_state=9)
        estimator.fit(data)
        s_score = metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size=5000,
                                           random_state=9)
        print('\tcluster:{:1d}  silhouette_score:{:.5f}'.format(cur_n_cluster, s_score))
        if s_score > max_s_score:
            n_cluster = cur_n_cluster
            max_s_score = s_score

    return n_cluster


def get_contact_ratio(result):
    """
    计算各簇推荐视频的相似度
    Args:
        result: 推荐结果
    Returns:
        contact_ratio: 视频相似度
    """
    cluster_list = list(result.keys())
    contact_ratio = list()
    for i in range(len(cluster_list)):
        for j in range(i + 1, len(cluster_list)):
            set_i = set(result[cluster_list[i]])
            set_j = set(result[cluster_list[j]])
            cur_ratio = 2 * len(set_i & set_j) / (len(set_i) + len(set_j))
            contact_ratio.append(cur_ratio)
    contact_ratio = np.mean(contact_ratio)
    return contact_ratio


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


def train_model(view_num=50, launch_num=5, recall_num=30):
    """
    生成模型及推荐结果
    Args:
        view_num: 每人至少观看的视频数量
        launch_num: 试探视频个数
        recall_num: 各类视频召回个数
    """

    ###########################
    log_data = get_log_data('20190425', 30)  # 拉取最近 30 天的 type=2 的日志
    video_tag = get_video_tag()  # 日志对应的视频 tag 信息
    log_data = log_data.merge(video_tag, on='video_id', how='inner')
    log_data = log_data[log_data['counter'] >= 90]

    # log_data 为最近 30 天 type=2 且 counter>=90 的用户观看日志
    # log_data 包含的特征 ['gaid', 'video_id', 'counter', 'tags']
    ###########################

    # 过滤用户
    user_count = log_data['gaid'].value_counts()
    uni_users = user_count[user_count >= view_num].index  # 获取观看大于 50 的 gaid
    log_data = log_data[log_data['gaid'].apply(lambda x: x in uni_users)]
    print('Unique Users:', len(uni_users))
    print('the Total Number of Logs:', log_data.shape[0])
    # 构造用户兴趣特征
    user_tag = dict()
    for cur_user, group in log_data.groupby('gaid'):
        temp_data = group['tags'].value_counts() ** 0.7
        cur_rate = temp_data / temp_data.sum()
        user_tag[cur_user] = cur_rate.to_dict()

    user_tag = pd.DataFrame(user_tag).T
    user_tag.fillna(0, inplace=True)
    # 通过轮廓系数寻找最优聚类个数
    k_cluster = get_cluster_num(user_tag)
    # 对用户进行聚类
    kmeans = KMeans(init='k-means++', n_clusters=k_cluster, n_init=10, random_state=9)
    pred_result = kmeans.fit_predict(user_tag)
    cluster_num = pd.Series(pred_result).value_counts().sort_index().to_dict()

    print("the Distribution of Cluster Members[{}]:\n\t{}".format(k_cluster, cluster_num))
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = pd.DataFrame(cluster_centers, columns=user_tag.columns)
    cluster_centers[cluster_centers < 0] = 0
    # 寻找最具区分性的 Tag
    launch_ratio, major_tags = get_launch_info(cluster_centers, launch_num)
    print('the Main Tags of Video:', ', '.join(major_tags))
    print('the Number of Different Tags:\n\t', launch_ratio)
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
    # 训练并保存决策树 pmml 格式
    x = user_tag.values.tolist()
    y = pred_result.tolist()
    pipeline = PMMLPipeline([("classifier", tree.DecisionTreeClassifier(max_depth=5))]);
    pipeline.fit(x, y)
    ########## 模型保存路劲 ##########
    model_path = r"C:\Users\tn_jinfeng.zhang\Desktop\DecisionTree.pmml"
    sklearn2pmml(pipeline, model_path, with_repr=True)
    # 计算各类推荐的视频
    user_cluster = pd.DataFrame({'gaid': user_tag.index, 'cluster': pred_result})
    log_data = log_data.merge(user_cluster, on='gaid')
    rec_result = dict()
    for c, group in log_data.groupby('cluster'):
        rec_result[c] = list(group['video_id'].value_counts().index[:recall_num])
    rec_info = {c: len(rec_result[c]) for c in rec_result.keys()}
    print('the Recommended Amounts:\n\t', rec_info)
    # 计算各类视频重合度
    contact_ratio = get_contact_ratio(rec_result)
    print('the Mean Contact Ratio: {:.2f}'.format(contact_ratio))
    # 保存各类推荐结果

    ########## 召回视频保存路劲 ##########
    with open(r"C:\Users\tn_jinfeng.zhang\Desktop\recall_videos.json", 'w') as f:
        json.dump(rec_result, f)
    # 保存试投标签
    ########## 试投类型保存路劲 ##########
    with open(r"C:\Users\tn_jinfeng.zhang\Desktop\launch_tags.json", 'w') as f:
        json.dump(launch_ratio, f)

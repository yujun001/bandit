
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import tree

# def get_cluster_num(data, limit_cluster=10):
#     print('the Clustering Results:')
#     n_cluster = 0
#     max_s_score = 0
#     for cur_n_cluster in range(2, limit_cluster + 1):
#         estimator = KMeans(init='k-means++',
#                            n_clusters=cur_n_cluster,
#                            n_init=10,
#                            random_state=9)
#         estimator.fit(data)
#         s_score = metrics.silhouette_score(data,
#                                            estimator.labels_,
#                                            metric='euclidean',
#                                            sample_size=5000,
#                                            random_state=9)
#         print('\tcluster:{:1d}  silhouette_score:{:.5f}'.format(cur_n_cluster, s_score))
#         if s_score > max_s_score:
#             n_cluster = cur_n_cluster
#             max_s_score = s_score
#
#     return n_cluster
#
#
# path = "/Users/jun_yu/Downloads/res_select.csv"
# df = pd.read_csv(path)
# k_cluster = get_cluster_num(df)
# print(k_cluster)

# path = "/Users/jun_yu/Downloads/part-00000-4a83a190-9378-4015-b29c-b920e09b56a0-c000.csv"
# df = pd.read_csv(path)
# del df['gaid']
#
# print(df.columns.tolist())
#
# res = dict()
# for col in df.columns.tolist():
#     # print(col)
#     val = df[df[col] !=0][col].count()
#     # print(val)
#     res[col] = val
#
# # print(res)
#
# res_df = pd.DataFrame(list(res.items()), columns=['type', 'num'])
# # print(res_df.head(100))
# res_df.to_csv("./res_df.csv", index=False)
#
# type_select = res_df[res_df['num'] > 10000]['type'].tolist()
# print(type_select)
# print(len(type_select))
#
# df_select = df[type_select]
# df_select.to_csv("./res_select.csv", index=False)
# print(df_select.T.head(100))



# for i in df

# dataMat = np.array(df)
# print(dataMat)
#
# #调用sklearn中的PCA，其中主成分有5列
# pca_sk = PCA(n_components=5)
# #利用PCA进行降维，数据存在newMat中
# newMat = pca_sk.fit_transform(dataMat)
# print(newMat)
#
# # kmeans = KMeans(n_clusters=3,random_state=0).fit(newMat)
#
# k_cluster = get_cluster_num(newMat)
# # print(k_cluster)




# df_rs = pd.read_csv("/home/yujun/Desktop/file_back/new_user_kmeans/df_num.csv")
#
# df_rs_tmp = df_rs[['brand', 'net_type', 'item_type','user_or_not']]
# df_rs_tmp = df_rs_tmp.drop_duplicates()
# df_rs = df_rs_tmp.as_matrix()
#
# print(df_rs)
# print(type(df_rs))
#
# df_scale = preprocessing.scale(df_rs)
#
# from sklearn.cluster import KMeans
# itreation = 500000000
# model = KMeans(n_clusters =3,n_jobs = 4,max_iter = itreation)
# model.fit(df_scale)
#
# y = model.labels_
# X = df_scale
#
# print(y)
# print(X)
#
# pca = PCA(n_components=2)
# X_p = pca.fit_transform(X)
# print(X_p)
#
# ax = plt.figure()
#
#
# #     # .transform(X)
# # ax = plt.figure()
# #
# # #x = np.arange(10)
# # #ys = [i+x+(i*x)**2 for i in range(10)]
# # #colors = cm.rainbow(np.linspace(0, 1, len(ys)))
# colors = 'rgb'
# target_names = list(['label_0','label_1','label_2'])
# for c, i, target_name in zip(colors, [0, 1, 2], target_names):
#     plt.scatter(X_p[y==i, 0], X_p[y==i, 1], c= c,label=target_name)
#
# plt.xlabel('Dimension1')
# plt.ylabel('Dimension2')
# plt.legend()
# plt.show()


import numpy as np
np.random.seed(0)
import seaborn as sns

sns.set()
uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data)


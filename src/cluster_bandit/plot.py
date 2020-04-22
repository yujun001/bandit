
import pandas as pd
import matplotlib.pyplot as plt

df_video = pd.read_csv('/Users/jun_yu/Downloads/numViewsPerVideo'
                       '/part-00000-e6c867f9-4136-474e-8fe4-247f446b89d7-c000.csv')

df_gaid = pd.read_csv('/Users/jun_yu/Downloads/numViewsPerGaid'
                      '/part-00000-abea7892-fd40-438a-9583-46d8cd7d2c20-c000.csv')

print(df_video.head(10))
print(df_gaid.head(10))

df = df_gaid
x = df['numViewsGaid']
plt.hist(x,
         range=(0,120),
         rwidth=0.6,
         bins=40,
         density=True,  #normed=True是频率图，默认是频数图
         weights=None,
         cumulative=False,
         bottom=None,
         histtype=u'bar',
         align=u'left',
         orientation=u'vertical',
         log=False,
         color=None,
         label=None,
         stacked=False
         )
plt.title("Frequency")
plt.xlabel('Travel Counter ')
plt.show()


df_y = df_video
y = df_y['numViewsVideo']
plt.hist(y,
         range=(0,500),
         rwidth=0.6,
         bins=40,
         density=True,  #normed=True是频率图，默认是频数图
         weights=None,
         cumulative=False,
         bottom=None,
         histtype=u'bar',
         align=u'left',
         orientation=u'vertical',
         log=False,
         color=None,
         label=None,
         stacked=False
         )
plt.title("Frequency")
plt.xlabel('Travel Counter ')
plt.show()


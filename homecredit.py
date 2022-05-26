import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import plot_tree

import seaborn as sns
import matplotlib.pyplot as plt

# Figure Size
plt.rcParams["figure.figsize"] = (10.0, 7.0)

app_train = pd.read_csv("../home-credit-default-risk/application_train.csv")
app_test = pd.read_csv("../home-credit-default-risk/application_test.csv")

#print(app_train.info())

#print(app_train.isnull().sum().sort_values())
"""
sns.histplot(
    app_train["AMT_CREDIT"],    # データ
    kde = True,      # 近似密度関数の表示有無
    bins = 100        # 変数の刻み数
) 

sns.histplot(
    app_train["AMT_INCOME_TOTAL"],    # データ
    kde = True,      # 近似密度関数の表示有無
    bins = 100        # 変数の刻み数
) 
"""
sns.histplot(
    app_train["AMT_GOODS_PRICE"],    # データ
    kde = True,      # 近似密度関数の表示有無
    bins = 100        # 変数の刻み数
).set_title("AMT_GOODS_PRICE")

plt.show()
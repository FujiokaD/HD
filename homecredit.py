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

sns.histplot(
    app_train["AMT_GOODS_PRICE"],    # データ
    kde = True,      # 近似密度関数の表示有無
    bins = 100        # 変数の刻み数
).set_title("AMT_GOODS_PRICE")

plt.show()
"""
#モデル作成
app_train.select_dtypes(include=object).columns.values

from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
encoded = oe.fit_transform(app_train[app_train.select_dtypes(include=object).columns.values])
decoded = oe.inverse_transform(encoded)

# 欠損値は一律-999で補完
train_x = app_train.drop(columns=["TARGET", "SK_ID_CURR"]).select_dtypes(include=[int, float]).fillna(-999)
train_y = app_train[["TARGET"]]

col_name = list(train_x.columns.values)

tree_model = DecisionTreeClassifier(
    criterion="gini",            # Entropy基準の場合は"entropy”
    splitter="best",             # 分割をランダムで行う場合は"random"
    random_state=17,             # 同じ分割スコアの時にランダムに選ぶseedを固定
    max_depth=5,                 # 決定木の深さの最大値
    min_samples_split=500,       # 分割する最小データ数
    min_samples_leaf=100         # 末端ノードに該当する最小サンプル数
)
tree_model = tree_model.fit(train_x, train_y)

# scikit-learn 0.21以降から実装された
fig = plt.figure(figsize=(40, 20))
ax = fig.add_subplot()
split_info = plot_tree(tree_model, feature_names=col_name, ax=ax, filled=True)
plt.show()

pred = tree_model.predict_proba(train_x)[:, 1]
print(roc_auc_score(train_y, pred))
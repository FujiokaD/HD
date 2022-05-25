import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import plot_tree

#import seaborn as sns
import matplotlib.pyplot as plt

# Figure Size
plt.rcParams["figure.figsize"] = (15.0, 10.0)

app_train = pd.read_csv("../home-credit-default-risk/application_train.csv")
app_test = pd.read_csv("../home-credit-default-risk/application_test.csv")

print(app_train.info())

print(app_train.isnull().sum().sort_values())
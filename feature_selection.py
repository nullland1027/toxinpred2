import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

df = pd.read_csv('out.out')  # Read aac file
X = np.array(df)
y = np.array([0, 1, 0, 1, 1, 0, 1])
l1_svc = LinearSVC(C=0.1, penalty='l1', dual=False).fit(X, y)  # L1正则化的SVC模型
model = SelectFromModel(l1_svc, prefit=True)  # 特征选择器
X_new = model.transform(X)  # 选择特征
print(pd.DataFrame(X_new))

# coding = uft-8

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('NBA_Season_Stats.csv')
df_C = df.loc[df['Pos'] == 'C']
df_PF = df.loc[df['Pos'] == 'PF']
df_PG = df.loc[df['Pos'] == 'PG']
df_SF = df.loc[df['Pos'] == 'SF']
df_SG = df.loc[df['Pos'] == 'SG']
df_C = df_C.fillna(df_C.mean())
df_PF = df_PF.fillna(df_PF.mean())
df_PG = df_PG.fillna(df_PG.mean())
df_SF = df_SF.fillna(df_SF.mean())
df_SG = df_SG.fillna(df_SG.mean())

df = pd.concat([df_C, df_PG, df_SG, df_PF, df_SF])
df = df.sort_values(by='Year')

X = df.drop(columns=['Year', 'Player', 'Pos', 'Tm', 'G', 'MP'])
y = df['Pos']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=True, test_size=0.2)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.fit_transform(y_test)


# config
# 0 自己写的，1 sklearn库cart，2 集成学习优化(提升树adaboost)，3 随机森林（bagging）,4 梯度提升决策树(GBDT), 5 XGBoost,6 BP net
choose =0

if choose==0:
    # 自己写的bayes
    # 数据分类 以5分类
    n_split = 5
    for feature in range(X_test.shape[1]):
        feature_list = [item[feature] for item in X_test]
        sort_feature_list = sorted(list(feature_list))
        split_ = [sort_feature_list[int(x * len(sort_feature_list) / (n_split))] for x in range(n_split)]

        # print(split_)
        # print(len(feature_list))
        # [-2.208827279477261, -0.9589191849238148, 0.29098890962963164, 1.540897004183078, 2.7908050987365245]
        for i, item in enumerate(feature_list):
            for j in range(n_split):
                if j == n_split - 1:
                    feature_list[i] = n_split - 1
                    break
                if item >= split_[j] and item < split_[j + 1]:
                    feature_list[i] = j
                    break
        for i,item in enumerate(X_test):
            item[feature]=int(feature_list[i])

    for feature in range(X_train.shape[1]):
        feature_list = [item[feature] for item in X_train]
        sort_feature_list = sorted(list(feature_list))
        split_ = [sort_feature_list[int(x * len(sort_feature_list) / (n_split))] for x in range(n_split)]

        # print(split_)
        # print(len(feature_list))
        # [-2.208827279477261, -0.9589191849238148, 0.29098890962963164, 1.540897004183078, 2.7908050987365245]
        for i, item in enumerate(feature_list):
            for j in range(n_split):
                if j == n_split - 1:
                    feature_list[i] = n_split - 1
                    break
                if item >= split_[j] and item < split_[j + 1]:
                    feature_list[i] = j
                    break
        for i,item in enumerate(X_train):
            item[feature]=int(feature_list[i])

    classes = np.unique(y_train)

    from model.C45 import DecisionTree
    model = DecisionTree()
    model.fit(X_train.tolist(), y_train_encoded,[int(i) for i in range(23)])
    y_pred = model.predict(X_test.tolist())
    y_test = y_test_encoded

if choose==1:
    #sklearn库
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(criterion='gini')
    model.fit(X_train,y_train_encoded)
    y_pred = model.predict(X_test)
    y_test = y_test_encoded

if choose==2:
    #集成学习优化 提升树
    from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='gini'))
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

if choose==3:
    #集成学习优化
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    model = RandomForestClassifier(criterion='gini')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

if choose==4:
    #集成学习优化
    from sklearn.ensemble import GradientBoostingClassifier
    # from sklearn.tree import DecisionTreeClassifier
    model = GradientBoostingClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

if choose==5:
    #集成学习优化
    from xgboost import XGBClassifier
    # from sklearn.tree import DecisionTreeClassifier
    model = XGBClassifier()
    model.fit(X_train,y_train_encoded)
    y_pred = model.predict(X_test)
    y_test = y_test_encoded

if choose==6:
    from sklearn.neural_network import MLPClassifier
    model=MLPClassifier(hidden_layer_sizes=(100,200))
    model.fit(X_train,y_train_encoded)
    y_pred=model.predict(X_test)
    y_test=y_test_encoded






# 可视化
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print('F1:', f1)
print('Precision:', precision)
print('Recall:', recall)

try:
    classes = np.unique(y_train)
    colors = ['#6894b9', '#c4c3be', '#edbaa7', '#ef9749', '#6e6e4b']
    f1 = f1_score(y_test, y_pred, average=None)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)


    # f1
    plt.figure()
    plt.bar(classes,f1.tolist(),color=colors)
    # plt.xticks
    for i in range(len(classes)):
        plt.text(x=classes[i], y=f1[i],s=round(f1[i],2),ha='center')
    plt.title('F1')
    plt.show()

    # precision
    plt.figure()
    plt.bar(classes,precision.tolist(),color=colors)
    # plt.xticks
    for i in range(len(classes)):
        plt.text(x=classes[i], y=precision[i],s=round(precision[i],2),ha='center')
    plt.title('Precision')
    plt.show()

    # recall
    plt.figure()
    plt.bar(classes,recall.tolist(),color=colors)
    # plt.xticks
    for i in range(len(classes)):
        plt.text(x=classes[i], y=recall[i],s=round(recall[i],2),ha='center')
    plt.title('Recall')
    plt.show()

except Exception as e:
    print(e)

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# df = pd.read_csv('NBA_Season_Stats.csv')
# df = df.fillna(0)

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

df = df[df['TOV']!=0]
df['ATR']=df['AST']/df['TOV']

# X = df[['3P','3PA']]
# X = df.drop(columns=['Year','Player','Pos','Tm','G','MP','FG','FGA','2P','2PA','eFG%','FT','FTA','TOV','PTS'])
X = df.drop(columns=['Year','Player','Pos','Tm','G','MP'])
y = df['Pos']

# print(y)
scaler = MinMaxScaler()
X_scaled=scaler.fit_transform(X)
y = y.values


X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,shuffle=True,test_size=0.2)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

#0 自己写，1 kmeans库，2 KNN
choose = 2

if choose==0:
    # 自己写的kmeans
    from model.kmeans import KMeans

    model = KMeans(n_clusters=5,max_iter=300)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    y_test = le.fit_transform(y_test)
    print(y_pred)

if choose==1:
    # sklearn库
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=5,max_iter=10000)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    y_test = le.fit_transform(y_test)

if choose==2:
    # KNN KNN（K-Nearest Neighbors）分类是一种监督学习算法，它可以用来解决分类问题。它的基本思想是：对于一个未知类别的样本，我们可以找到训练集中与它最近的 K 个样本，然后根据这 K 个样本的类别来预测未知样本的类别。
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=35)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)



# BP神经网络
# from sklearn.neural_network import MLPClassifier
# model = MLPClassifier(hidden_layer_sizes=(100,200),)
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)


 # 可视化
try:
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print('F1:', f1)
    print('Precision:', precision)
    print('Recall:', recall)

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

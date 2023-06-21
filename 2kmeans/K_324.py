import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# df = pd.read_csv('NBA_Season_Stats.csv')
# df = df.fillna(0)

df = pd.read_csv('detect_resnet_324.csv')

# X = df.drop(columns=['Year','Player','Pos','Tm','G','MP','FG','FGA','2P','2PA','eFG%','FT','FTA','TOV','PTS'])
# X = df[['FG%','3P','3PA','3P%','FT%','ORB','TRB','AST','BLK']]
X = df.drop(columns=['Pos'])
y = df['Pos']



X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)

from sklearn.cluster import KMeans

model = KMeans(n_clusters=5, max_iter=10000)
model.fit(X_train)
y_pred = model.predict(X_test)

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
    plt.bar(classes, f1.tolist(), color=colors)
    # plt.xticks
    for i in range(len(classes)):
        plt.text(x=classes[i], y=f1[i], s=round(f1[i], 2), ha='center')
    plt.title('F1')
    plt.show()

    # precision
    plt.figure()
    plt.bar(classes, precision.tolist(), color=colors)
    # plt.xticks
    for i in range(len(classes)):
        plt.text(x=classes[i], y=precision[i], s=round(precision[i], 2), ha='center')
    plt.title('Precision')
    plt.show()

    # recall
    plt.figure()
    plt.bar(classes, recall.tolist(), color=colors)
    # plt.xticks
    for i in range(len(classes)):
        plt.text(x=classes[i], y=recall[i], s=round(recall[i], 2), ha='center')
    plt.title('Recall')
    plt.show()

except Exception as e:
    print(e)

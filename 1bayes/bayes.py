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

# X = df.drop(columns=['Year','Player','Pos','Tm','G','MP','FG','FGA','2P','2PA','eFG%','FT','FTA','TOV','PTS'])
# X = df[['FG%','3P','3PA','3P%','FT%','ORB','TRB','AST','BLK']]
X = df.drop(columns=['Year','Player','Pos','Tm','G','MP'])
y = df['Pos']

scaler = MinMaxScaler()
X_scaled=scaler.fit_transform(X)
y = y.values

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,shuffle=True,test_size=0.2)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# config 0 自己写的； 1 sklearn高斯贝叶斯； 2多项式贝叶斯； 3伯努利贝叶斯； 4集成学习
choose = 2

if choose==0:
    # 自己写的bayes
    from model.Bayes_model import NaiveBayes

    model = NaiveBayes()
    model.fit(X_train,y_train_encoded)
    y_pred = le.inverse_transform(model.predict(X_test))

if choose==1:
    # 高斯贝叶斯
    from sklearn.naive_bayes import GaussianNB

    model = GaussianNB()
    model.fit(X_train,y_train_encoded)
    y_pred = le.inverse_transform(model.predict(X_test))

if choose==2:
    # 多项式贝叶斯
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

if choose==3:
    # 伯努利贝叶斯
    from sklearn.naive_bayes import BernoulliNB
    model = BernoulliNB()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

if choose==4:
    # 集成学习
    from sklearn.ensemble import BaggingClassifier
    from sklearn.naive_bayes import MultinomialNB
    model = BaggingClassifier(base_estimator=MultinomialNB(),n_estimators=50,random_state=0)
    model.fit(X_train,y_train_encoded)
    y_pred = model.predict(X_test)
    y_pred=le.inverse_transform(y_pred)



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

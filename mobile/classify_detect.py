import warnings
warnings.filterwarnings('ignore')
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from paddlex import transforms as T
import paddlex as pdx
import paddle
import numpy as np
import pandas as pd
import shutil
import imghdr
from PIL import Image
from model.neural import Classifier

model = Classifier()
param_state_dict = paddle.load('my_best_model.pdparams')
model.set_state_dict(param_state_dict)

test_dataset = pd.read_csv('test.csv')
test_data = test_dataset.values.astype(np.float32)
test_data = test_data.reshape(-1, 1, 562)



y_pred = model(paddle.to_tensor(test_data[:, :, :-1]))
y_test = paddle.to_tensor(test_data[:,:,-1:])
y_pred = y_pred.argmax(1).numpy()
y_test=y_test.flatten().astype(int).numpy()

y_test= y_test.tolist()
y_pred=y_pred.tolist()

# 可视化
try:
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print('F1:', f1)
    print('Precision:', precision)
    print('Recall:', recall)


    classes = ['LAYING','STANDING','SITTING','WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS']
    colors = ['#6894b9', '#c4c3be', '#edbaa7', '#ef9749', '#6e6e4b']
    f1 = f1_score(y_test, y_pred, average=None)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)


    # f1
    plt.figure()
    plt.bar(classes,f1,color=colors)
    # plt.xticks
    for i in range(len(classes)):
        plt.text(x=classes[i], y=f1[i],s=round(f1[i],2),ha='center')
    plt.xticks(rotation=-12)

    plt.title('F1')
    plt.savefig('f1.png')

    # precision
    plt.figure()
    plt.bar(classes,precision,color=colors)
    # plt.xticks
    for i in range(len(classes)):
        plt.text(x=classes[i], y=precision[i],s=round(precision[i],2),ha='center')
    plt.xticks(rotation=-12)

    plt.title('Precision')
    plt.show()

    # recall
    plt.figure()
    plt.bar(classes,recall,color=colors)
    # plt.xticks
    for i in range(len(classes)):
        plt.text(x=classes[i], y=recall[i],s=round(recall[i],2),ha='center')
    plt.xticks(rotation=-12)

    plt.title('Recall')
    plt.show()

except Exception as e:
    print(e)
